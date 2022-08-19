import os
import json
from itertools import chain
from tqdm import tqdm
import logging
import data.spider.evaluation as evaluation
import data.spider.evaluation_new as evaluation_new


special_tokens = []
kmaps = evaluation.build_foreign_key_map_from_json("./data/spider/data/tables.json")
db_dir = "./data/spider/data/database/"

with open("./data/spider/data/dev_gold.sql") as f:
    gold_sql = [l.strip().split('\t') for l in f.readlines() if len(l.strip()) > 0]

def extra_key_semantics(sql):
    info = []
    for item in sql['select'][1]:
        i = item
        while not isinstance(i, str):
            i = i[1]
        info.append(i)

    for item in sql['from']['table_units']:
        if isinstance(item[1], str):
            info.append(item[1])
        else:
            info.extend(extra_key_semantics(item[1]))

    for item in sql['where']:
        if isinstance(item, tuple):
            i = item[2]
            while not isinstance(i, str):
                i = i[1]
            info.append(i)
            # i = item[3]
            # if not isinstance(i, str):
            #     print(sql)
            # info.append("value:" + i)

    if sql['groupBy'] is not None:
        for item in sql['groupBy']:
            i = item
            while not isinstance(i, str):
                i = i[1]
            info.append(i)
    
    if sql['orderBy'] is not None and len(sql['orderBy']) > 0:
        for item in sql['orderBy'][1]:
            i = item
            while not isinstance(i, str):
                i = i[1]
            info.append(i)
    
    if sql['having'] is not None and len(sql['having']) > 0:
            for item in sql['having']:
                if isinstance(item, tuple):
                    i = item[2]
                    while not isinstance(i, str):
                        i = i[1]
                    info.append(i)
                    # i = item[3]
                    # if not isinstance(i, str):
                    #     print(sql)
                    # info.append("value:" + i)
    
    if sql['union'] is not None:
        info.extend(extra_key_semantics(sql['union']))
    
    if sql['intersect'] is not None:
        info.extend(extra_key_semantics(sql['intersect']))
    
    if sql['except'] is not None:
        info.extend(extra_key_semantics(sql['except']))
    
    return list(filter(("*").__ne__, info))

def load_data(args):
    print('Load questions')
    train_set = json.load(open(os.path.join(args.input_dir, 'spider_train.json')))
    val_set = json.load(open(os.path.join(args.input_dir, 'spider_eval.json')))
    test_set = json.load(open(os.path.join(args.input_dir, 'spider_eval.json')))
    for question in chain(train_set, val_set, test_set):
        try:
            db_name = question['db_id']
            db_path = os.path.join(db_dir, db_name, db_name + ".sqlite")
            schema = evaluation.Schema(evaluation.get_schema(db_path))
            sql = evaluation.get_sql(schema, question['query'])
            key_info = extra_key_semantics(sql)
        except:
            question['input'] = None
            question['target'] = None
            question['ir'] = None
            continue
        
        question['input'] = "question: {} ; database: {}".format(question['text_in'], question['struct_in'].lower())
        question['target'] = question['query'].lower()
        
        key_info_seq = ""
        for item in key_info:
            item_split = item.split(".")
            if len(item_split) > 1:
                item = item_split[-1]
            key_info_seq += "<A> {} </A> ".format(item)
        question['ir'] = key_info_seq.strip()
        
    return train_set, val_set, test_set

def evaluate(args, outputs, targets, *xargs):
    correct = 0
    exec_correct = 0
    evaluator = evaluation.Evaluator()
    for pred, gold, real in tqdm(zip(outputs, targets, gold_sql), total=len(outputs)):
        if pred.lower() == gold.lower():
            correct += 1
            exec_correct += 1
            continue
        g_sql_str, db_name = real
        db_path = os.path.join(db_dir, db_name, db_name + ".sqlite")
        schema = evaluation.Schema(evaluation.get_schema(db_path))
        g_sql = evaluation.get_sql(schema, g_sql_str)
        hardness = evaluator.eval_hardness(g_sql)
        try:
            p_sql = evaluation.get_sql(schema, pred)
        except:
            continue
        
        # rebuild sql for value evaluation
        kmap = kmaps[db_name]
        g_valid_col_units = evaluation.build_valid_col_units(g_sql['from']['table_units'], schema)
        g_sql = evaluation.rebuild_sql_val(g_sql)
        g_sql = evaluation.rebuild_sql_col(g_valid_col_units, g_sql, kmap)
        p_valid_col_units = evaluation.build_valid_col_units(p_sql['from']['table_units'], schema)
        p_sql = evaluation.rebuild_sql_val(p_sql)
        p_sql = evaluation.rebuild_sql_col(p_valid_col_units, p_sql, kmap)
        
        try:
            exec_score = evaluation.eval_exec_match(db_path, pred, g_sql_str, p_sql, g_sql)
            exec_correct += int(exec_score)
        except:
            continue

        try:
            exact_score = evaluator.eval_exact_match(p_sql, g_sql)
            correct += int(exact_score)
        except:
            continue

    logging.info("Execution: {}, Exact Match: {}".format(exec_correct / len(outputs), correct / len(outputs)))

    return correct / len(outputs)

def compute_exact_match_metric(predictions, references):
    foreign_key_maps = dict()
    for reference in references:
        if reference["db_id"] not in foreign_key_maps:
            foreign_key_maps[reference["db_id"]] = evaluation_new.build_foreign_key_map(
                {
                    "table_names_original": reference["db_table_names"],
                    "column_names_original": list(
                        zip(
                            reference["db_column_names"]["table_id"],
                            reference["db_column_names"]["column_name"],
                        )
                    ),
                    "foreign_keys": list(
                        zip(
                            reference["db_foreign_keys"]["column_id"],
                            reference["db_foreign_keys"]["other_column_id"],
                        )
                    ),
                }
            )
    evaluator = evaluation_new.Evaluator(db_dir, foreign_key_maps, "match")
    for prediction, reference in zip(predictions, references):
        turn_idx = reference.get("turn_idx", 0)
        # skip final utterance-query pairs
        if turn_idx < 0:
            continue
        _ = evaluator.evaluate_one(reference["db_id"], reference["query"], prediction)
    evaluator.finalize()
    return {
        "exact_match": evaluator.scores["all"]["exact"],
    }

def compute_test_suite_metric(predictions, references):
    foreign_key_maps = dict()
    for reference in references:
        if reference["db_id"] not in foreign_key_maps:
            foreign_key_maps[reference["db_id"]] = evaluation_new.build_foreign_key_map(
                {
                    "table_names_original": reference["db_table_names"],
                    "column_names_original": list(
                        zip(
                            reference["db_column_names"]["table_id"],
                            reference["db_column_names"]["column_name"],
                        )
                    ),
                    "foreign_keys": list(
                        zip(
                            reference["db_foreign_keys"]["column_id"],
                            reference["db_foreign_keys"]["other_column_id"],
                        )
                    ),
                }
            )

    evaluator = evaluation_new.Evaluator(
        db_dir=db_dir,
        kmaps=foreign_key_maps,
        etype="exec",
        plug_value=False,
        keep_distinct=False,
        progress_bar_for_each_datapoint=False,
    )
    # Only used for Sparc/CoSQL
    # turn_scores = {"exec": [], "exact": []}
    # for prediction, reference in zip(predictions, references):
    #     turn_idx = reference.get("turn_idx", 0)
    #     # skip final utterance-query pairs
    #     if turn_idx < 0:
    #         continue
    #     try:
    #         _ = evaluator.evaluate_one(
    #             reference["db_id"],
    #             reference["query"],
    #             prediction,
    #             turn_scores,
    #             idx=turn_idx,
    #         )
    #     except AssertionError as e:
    #         logging.warning(f"unexpected evaluation error: {e.args[0]}")
    evaluator.finalize()
    return {
        "exec": evaluator.scores["all"]["exec"],
    }

