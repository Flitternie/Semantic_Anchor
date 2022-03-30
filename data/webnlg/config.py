import os

special_tokens = ['<H>', '</H>', '<R>', '</R>' '<T>', '</T>', '<S>']

def reorder_triples(sequences):
    reordered_sequences = []
    for sequence in sequences:
        triples = sequence.strip().split('<H>')
        triples = [triple for triple in triples if triple != '']
        triples = [(triple.split('<R>')[0].strip(), triple.split('<R>')[1].split('<T>')[0].strip(), triple.split('<T>')[-1].strip()) for triple in triples]
        reorder = {}
        for triple in triples:
            if triple[0] not in reorder:
                reorder[triple[0]] = {triple[1]: [triple[2]]}
            else:
                if triple[1] not in reorder[triple[0]]:
                    reorder[triple[0]][triple[1]] = [triple[2]]
                else:
                    reorder[triple[0]][triple[1]].append(triple[2])
        reordered_sequence = ''
        for key, value in reorder.items():
            reordered_sequence += " <S> <H> {} </H> ".format(key)
            for k, v in value.items():
                reordered_sequence += " <R> {} </R> <T> {} </T> ".format(k, ' ; '.join(v))
            reordered_sequence += ' </S> '
        reordered_sequences.append(reordered_sequence.replace('>  <', '> <').strip())
    return reordered_sequences

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
    
    print('Reorder triples')
    print(train_source[-1])
    train_source = reorder_triples(train_source)
    val_source = reorder_triples(val_source)
    test_source = reorder_triples(test_source)
    print(train_source[-1])
    
    train_set = [{'input': s, 'target': t} for s, t in zip(train_source, train_target)]
    val_set = [{'input': s, 'target': t} for s, t in zip(val_source, val_target)]
    test_set = [{'input': s, 'target': t} for s, t in zip(test_source, test_target)] 
    
    return train_set, val_set, test_set, vocab