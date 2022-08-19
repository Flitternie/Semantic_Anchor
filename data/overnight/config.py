import json
import os
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from data.overnight.evaluator.domain_base import Domain

special_tokens = []
overnight_domains = ['basketball', 'blocks', 'calendar', 'housing', 'publications', 'recipes', 'restaurants', 'socialnetwork']

def load_data(args):
    print('Load questions')
    train_set = json.load(open(os.path.join(args.input_dir, 'train.json')))
    val_set = json.load(open(os.path.join(args.input_dir, 'val.json')))
    test_set = json.load(open(os.path.join(args.input_dir, 'test.json')))
    return train_set, val_set, test_set

def evaluate(args, outputs, targets, all_extra_ids, *xargs):
    assert len(outputs) == len(targets)
    data = [[[],[]] for _ in range(len(all_extra_ids))]
    evaluators = [Domain.from_dataset(domain) for domain in overnight_domains]
    for p, g, d in zip(outputs, targets, all_extra_ids):
        data[d][0].append(p)
        data[d][1].append(g)
    scores = []
    for i, evaluator in enumerate(evaluators):
        domain_score = evaluator.compare_logical_form(data[i][0], data[i][1])
        scores += domain_score
        logging.info("{}-domain accuracy: {}".format(overnight_domains[i], np.mean(domain_score)))
    return np.mean(scores)


