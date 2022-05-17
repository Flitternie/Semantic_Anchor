import random
import os
import pandas as pd

special_tokens = []
overnight_domains = ['basketball', 'blocks', 'calendar', 'housing', 'publications', 'recipes', 'restaurants', 'socialnetwork']

def load_data(args):
    print('Build kb vocabulary')
    vocab = {
        'answer_token_to_idx': {}
    }

    print('Load questions')
    train_set, val_set, test_set = [], [], []
    
    for domain in overnight_domains:
        idx = overnight_domains.index(domain)
        train_data = read_overnight(os.path.join(args.input_dir, domain + '_train.tsv'), idx)
        random.shuffle(train_data)
        train_set += train_data[:int(len(train_data)*0.8)]
        val_set += train_data[int(len(train_data)*0.8):]
        test_set += read_overnight(os.path.join(args.input_dir, domain + '_test.tsv'), idx)
    
    return train_set, val_set, test_set, vocab

def read_overnight(path, domain_idx):
    ex_list = []
    infile = pd.read_csv(path, sep='\t')
    for idx, row in infile.iterrows():
        ex_list.append({'input': row['utterance'].strip(), 'target': row['logical_form'].strip(), 'ir': row['original'].strip(), 'domain': domain_idx})
    return ex_list


