import os
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