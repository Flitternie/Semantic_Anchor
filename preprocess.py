import os
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import re
import regex
import random

from utils.misc import init_vocab
from transformers import *
from utils.data import load_general, load_kqapro

def get_program_seq(program):
    seq = []
    for item in program:
        func = item['function']
        inputs = item['inputs']
        seq.append(func + '(' + '<c>'.join(inputs) + ')')
    seq = '<b>'.join(seq)
    return seq

def encode_dataset(mode, dataset, vocab, tokenizer):
    inputs = []
    targets = []
    choices = []
    answers = []
    
    for item in tqdm(dataset):
        inputs.append(item['input'])
        targets.append(item['output'])
        if 'choices' in item.keys():
            choices.append([vocab['answer_token_to_idx'][w] for w in item['choices']])
        if 'answers' in item.keys():
            answers.append(vocab['answer_token_to_idx'].get(item['answer']))
        
    sequences = inputs + targets
    encoded_inputs = tokenizer(sequences, padding = True)
    
    max_seq_length = len(encoded_inputs['input_ids'][0])
    assert max_seq_length == len(encoded_inputs['input_ids'][-1])

    input_ids = tokenizer.batch_encode_plus(inputs, max_length = max_seq_length, padding='max_length', truncation = True)
    source_ids = np.array(input_ids['input_ids'], dtype = np.int32)
    source_mask = np.array(input_ids['attention_mask'], dtype = np.int32)
    
    target_ids = tokenizer.batch_encode_plus(targets, max_length = max_seq_length, padding='max_length', truncation = True)
    target_ids = np.array(target_ids['input_ids'], dtype = np.int32)
    
    choices = np.array(choices, dtype = np.int32) if choices else np.array([0]*len(inputs), dtype = np.int32)
    answers = np.array(answers) if answers else np.array([0]*len(inputs), dtype = np.int32)
    
    return source_ids, source_mask, target_ids, choices, answers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model_name_or_path', required=True)
    args = parser.parse_args()
    set_seed(666)

    train_set, val_set, test_set, vocab = load_general(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    fn = os.path.join(args.output_dir, 'vocab.json')
    print('Dump vocab to {}'.format(fn))

    with open(fn, 'w') as f:
        json.dump(vocab, f, indent=2)
    
    tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)
    
    for name, dataset in zip(('train', 'val', 'test'), (train_set, val_set, test_set)):
        print('Encode {} set'.format(name))
        outputs = encode_dataset(args.mode, dataset, vocab, tokenizer)
        assert len(outputs) == 5
        with open(os.path.join(args.output_dir, '{}.pt'.format(name)), 'wb') as f:
            for o in outputs:
                pickle.dump(o, f)

if __name__ == '__main__':
    main()