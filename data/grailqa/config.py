import json
from itertools import chain
import os

special_tokens = ['AND', 'COUNT', 'JOIN', 'ARGMAX', 'ARGMIN', 'LT', 'LE', 'GT', 'GE']

def load_data(args):
    print('Build kb vocabulary')
    vocab = {
        'answer_token_to_idx': {}
    }

    print('Load questions')
    train_set = json.load(open(os.path.join(args.input_dir, 'grailqa_v1.0_train.json')))
    val_set = json.load(open(os.path.join(args.input_dir, 'grailqa_v1.0_dev.json')))
    # test_set = json.load(open(os.path.join(args.input_dir, 'grailqa_v1.0_test_public.json')))
    for question in chain(train_set, val_set):
        question['input'] = question.pop('question')
        question['target'] = question.pop('s_expression')
    return train_set, val_set, val_set, vocab
