import re
import numpy as np

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
    



