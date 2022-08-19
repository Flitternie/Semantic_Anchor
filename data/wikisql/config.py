import os
import json
from itertools import chain
from tqdm import tqdm
from data.wikisql.dbengine import DBEngine

special_tokens = []
agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
cond_ops = ['=', '>', '<', 'OP']

def load_data(args):
    print('Load questions')
    train_set = json.load(open(os.path.join(args.input_dir, 'train.json')))
    val_set = json.load(open(os.path.join(args.input_dir, 'dev.json')))
    test_set = json.load(open(os.path.join(args.input_dir, 'test.json')))
    for question in chain(train_set, val_set, test_set):
        question['input'] = "question : {} ; database : {}".format(question['text_in'], " </s> ".join(question['table']['header']).lower())
        key_info = "<A> {} </A>".format(question['table']['header'][question['sql']['sel']]) 
        for i in range(len(question['sql']['conds']['column_index'])):
            key_info += " <A> {} </A>".format(question['table']['header'][question['sql']['conds']['column_index'][i]].lower())
        question['ir'] = key_info
        question['target'] = get_logical_form(question)
    return train_set, val_set, test_set

def evaluate(args, outputs, targets, *xargs):
    correct = 0
    data_split = 'test' if args.inference else 'dev'
    db = DBEngine('./data/wikisql/data/{}.db'.format(data_split))
    with open('data/wikisql/data/{}.json'.format(data_split), 'r') as f:
        data = json.load(f)
    for pred, gold, d in tqdm(zip(outputs, targets, data), total=len(outputs)):
        if pred.lower() == gold.lower():
            correct += 1
            continue
        try:
            pred_sql = db.generate_logical_form(args.tokenizer, pred, d['question'], d['table'], agg_ops, cond_ops, 'sql')
        except:
            continue
        gold_sql = db.generate_logical_form(args.tokenizer, gold, d['question'], d['table'], agg_ops, cond_ops, 'sql')
        try:
            pred_ans = process_value(db.execute_query(d['table']['id'], pred_sql))
        except:
            continue
        gold_answer = process_value(db.execute_query(d['table']['id'], gold_sql))
        if pred_ans == gold_answer:
            correct += 1
    return correct / len(outputs)

def extract(item):
    split_sql = item.split('WHERE')
    if len(split_sql) < 2:
        return item.strip(), []
    elif len(split_sql) == 2:
        target, constraints = split_sql[0].strip(), split_sql[1].strip()
        constraints = constraints.split('AND')
        constraints = [constraint.strip().lower() for constraint in constraints if constraint != '']
        constraints.sort()
        return target, constraints
    else:
        return None, None

def process_value(values):
    processed_values = []
    for v in set(values):
        if isinstance(v, (list, tuple)):
            v = v[0]
        if isinstance(v, int):
            processed_values.append(str(float(v)))
        elif isinstance(v, float):
            processed_values.append(str(v))
        elif isinstance(v, str):
            try:
                processed_values.append(str(float(v)))
            except:
                processed_values.append(v.lower().strip())
        else:
            processed_values.append(v)
    processed_values.sort()
    return processed_values

def get_logical_form(raw_data):
    sql = raw_data['sql']
    columns = raw_data['table']['header']
    
    # generate label - sql statement
    sql_statement = 'SELECT ' + agg_ops[sql['agg']]
    if sql['agg'] > 0:
        sql_statement += '([' +  columns[sql['sel']] + ']) FROM table '
    else:
        sql_statement += ' [' +  columns[sql['sel']] + '] FROM table '

    if len(sql['conds']) > 0:
        sql_statement += 'WHERE '
        
        for i in range(len(sql['conds']['column_index'])):
            sql_statement += '[' + columns[sql['conds']['column_index'][i]] + '] ' + cond_ops[sql['conds']['operator_index'][i]]
            if isinstance(sql['conds']['condition'][i], (int, float)):
                sql_statement += " " + str(sql['conds']['condition'][i])
            else:
                sql_statement += " '" + sql['conds']['condition'][i] + "'"
            sql_statement += " AND "
        sql_statement = sql_statement[:-4]
            
    sql_statement = sql_statement.lower()

    return sql_statement