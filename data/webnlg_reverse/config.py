import os
import re
import numpy as np
from data.webnlg.config import reorder_triples

special_tokens = ['<H>', '</H>', '<R>', '</R>', '<T>', '</T>', '<S>']

def load_data(args):
    vocab = {
        'answer_token_to_idx': {}
    }
    print('Load data')
    train_source = open(os.path.join(args.input_dir, 'train.source')).readlines()
    train_target = open(os.path.join(args.input_dir, 'train.target')).readlines()
    val_source = open(os.path.join(args.input_dir, 'val.source')).readlines()
    val_target = open(os.path.join(args.input_dir, 'val.target')).readlines()
    test_source = open(os.path.join(args.input_dir, 'test_both.source')).readlines()
    test_target = open(os.path.join(args.input_dir, 'test_both.target')).readlines()
    
    if args.reorder:
        print('Reorder triples')
        train_source = reorder_triples(train_source) 
        val_source = reorder_triples(val_source)
        test_source = reorder_triples(test_source)
    
    train_set = [{'input': t, 'target': s} for s, t in zip(train_source, train_target)]
    val_set = [{'input': t, 'target': s} for s, t in zip(val_source, val_target)]
    test_set = [{'input': t, 'target': s} for s, t in zip(test_source, test_target)] 
    
    return train_set, val_set, test_set, vocab

def extract_triples(sequence):
    triples = []
    try:
        for triple in sequence.split('<H>'):
            if triple != '':
                triple = triple.split('<R>')
                triple = (triple[0].strip(), triple[1].split('<T>')[0].strip(), triple[1].split('<T>')[1].strip())
                triples.append(triple)
    except:
        pass
    return triples

def post_process(sequence):
    processed_sequence = ""
    try:
        reordered_triples = re.findall(r'''<H>(.*?)</H>''', sequence)
        for triple in reordered_triples:
            head = triple.strip().split('<R>')[0]
            for subtriple in re.findall(r'''<R>(.*?)</R>''', triple):
                relation = subtriple.strip().split('<T>')[0]
                tails = re.findall(r'''<T>(.*?)</T>''', subtriple)[0]
                for tail in tails.split('<S>'):
                    processed_sequence += " <H> {} <R> {} <T> {}".format(head.strip(), relation.strip(), tail.strip())
    except:
        pass
    return processed_sequence.strip()
    

def eval_accuracy(preds, golds):
    recall = []
    precision = []
    for pred, gold in zip(preds, golds):
        pred_triples = extract_triples(pred)
        gold_triples = extract_triples(gold)
        recall.append(np.mean([1 if gold_triple in pred_triples else 0 for gold_triple in gold_triples]))
        precision.append(np.mean([1 if pred_triple in gold_triples else 0 for pred_triple in pred_triples]))
    f1 = [2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) for i in range(len(precision))]
    return np.mean(recall), np.mean(precision), np.mean(f1)
        

def evaluate(args, outputs, targets):
    if args.reorder:
        outputs = [post_process(output) for output in outputs]
        targets = [post_process(target) for target in targets]
    recall, precision, f1 = eval_accuracy(outputs, targets)
    print("Recall: {}".format(recall))
    print("Precision: {}".format(precision))
    print("F1: {}".format(f1))
    return recall
    