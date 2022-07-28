import os
import json
from itertools import chain
from tqdm import tqdm
from data.wikisql.utils import DBEngine, Query, count_lines

special_tokens = []

def load_data(args):
    vocab = {
        'answer_token_to_idx': {}
    }

    print('Load questions')
    train_set = json.load(open(os.path.join(args.input_dir, 'train.json')))
    val_set = json.load(open(os.path.join(args.input_dir, 'dev.json')))
    test_set = json.load(open(os.path.join(args.input_dir, 'test.json')))
    for question in chain(train_set, val_set, test_set):
        question['input'] = "{} ; columns: {}".format(question['text_in'], " | ".join(question['table']['header']))
        # question['input'] = question['text_in']
        key_info = "<A> {} </A>".format(question['table']['header'][question['sql']['sel']]) 
        for i in range(len(question['sql']['conds']['column_index'])):
            # key_info += " <A> {} </A> <V> {} </V>".format(question['table']['header'][question['sql']['conds']['column_index'][i]], question['sql']['conds']['condition'][i])
            key_info += " <A> {} </A>".format(question['table']['header'][question['sql']['conds']['column_index'][i]])
        question['ir'] = key_info
        question['target'] = question['sql']['human_readable']
    return train_set, val_set, test_set, vocab

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

def evaluate(args, outputs, targets):
    correct = 0
    assert len(outputs) == len(targets)
    for pred, gold in zip(outputs, targets):
        if pred.lower() == gold.lower():
            correct += 1
            continue
        else:
            pred_target, pred_constraints = extract(pred)
            gold_target, gold_constraints = extract(gold)
            if pred_target is None or gold_target is None:
                continue
            if pred_target.lower() == gold_target.lower() and set(pred_constraints) == set(gold_constraints):
                correct += 1
    return correct / len(outputs)

