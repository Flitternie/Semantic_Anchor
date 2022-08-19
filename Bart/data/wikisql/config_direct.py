import os
import json
from itertools import chain

special_tokens = []

def load_data(args):
    print('Load questions')
    train_set = json.load(open(os.path.join(args.input_dir, 'wikisql_train.json')))
    val_set = json.load(open(os.path.join(args.input_dir, 'wikisql_eval.json')))
    test_set = json.load(open(os.path.join(args.input_dir, 'wikisql_test.json')))
    for question in chain(train_set, val_set, test_set):
        question['input'] = "{}; structed knowledge:  {}".format(question['text_in'], question['struct_in'])
        question['target'] = question['seq_out']
    return train_set, val_set, test_set

def evaluate(args, outputs, targets, *xargs):
    correct = 0
    for pred, gold in zip(outputs, targets):
        if pred.lower() == gold.lower():
            correct += 1
    return correct / len(outputs)



