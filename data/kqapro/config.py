import os
import re
import json
from itertools import chain
from tqdm import tqdm
from datetime import date
from data.kqapro.utils.load_kb import DataForSPARQL
from data.kqapro.utils.sparql_engine import get_sparql_answer

special_tokens = ['?', '^', '<', '>', '.', '"', '{', '}', ':', ';', '|', '!', '@', '#', '$', '%', '&', '*', '(', ')', '-', '_', '+', '=', '`', '~', '\'', ',']

def load_data(args):
    print('Build kb vocabulary')
    vocab = {
        'answer_token_to_idx': {}
    }
    print('Load questions')
    train_set = json.load(open(os.path.join(args.input_dir, 'train.json')))
    val_set = json.load(open(os.path.join(args.input_dir, 'val.json')))
    test_set = json.load(open(os.path.join(args.input_dir, 'test.json')))
    for question in chain(train_set, val_set, test_set):
        if not question['answer'] in vocab['answer_token_to_idx']:
            vocab['answer_token_to_idx'][question['answer']] = len(vocab['answer_token_to_idx'])
        question['input'] = question.pop('rewrite')
        question['ir'] = question['ir']
        question['target'] = question.pop('sparql')
        question['extra_id'] = vocab['answer_token_to_idx'].get(question['answer'])
    return train_set, val_set, test_set, vocab

def evaluate(args, outputs, targets, all_extra_ids, data):
    kb = DataForSPARQL(os.path.join("./data/kqapro/data/", 'kb.json'))
    given_answer = [[data.vocab['answer_idx_to_token'][int(a)] for a in list(al)] for al in all_extra_ids]
    correct = 0
    for ans, pred, gold in tqdm(zip(given_answer, outputs, targets), total=len(outputs)):
        if pred == gold:
            correct += 1
            continue
        pred_answer = get_sparql_answer(pred, kb)
        if pred_answer is None:
            pred_answer = 'no'
        is_match = whether_equal(ans[0], pred_answer)
        if is_match:
            correct += 1
    return correct / len(outputs)

def whether_equal(answer, pred):
    """
    check whether the two arguments are equal as attribute value
    """
    def truncate_float(x):
        # convert answer from '100.0 meters' to '100 meters'
        try:
            v, *u = x.split()
            v = float(v)
            if v - int(v) < 1e-5:
                v = int(v)
            if len(u) == 0:
                x = str(v)
            else:
                x = '{} {}'.format(str(v), ' '.join(u))
        except:
            pass
        return x

    def equal_as_date(x, y):
        # check whether x and y are equal as type of date or year
        try:
            x_split = x.split('-')
            y_split = y.split('-')
            if len(x_split) == 3:
                x = date(int(x_split[0]), int(x_split[1]), int(x_split[2]))
            else:
                x = int(x)
            if len(y_split) == 3:
                y = date(int(y_split[0]), int(y_split[1]), int(y_split[2]))
            else:
                y = int(y)
            if isinstance(x, date) and isinstance(y, date):
                return x == y
            else:
                x = x.year if isinstance(x, date) else x
                y = y.year if isinstance(y, date) else y
                return x == y
        except:
            return False

    answer = truncate_float(answer)
    pred = truncate_float(pred)
    if equal_as_date(answer, pred):
        return True
    else:
        return answer == pred

def post_process(text):
    pattern = re.compile(r'".*?"')
    nes = []
    for item in pattern.finditer(text):
        nes.append((item.group(), item.span()))
    pos = [0]
    for name, span in nes:
        pos += [span[0], span[1]]
    pos.append(len(text))
    assert len(pos) % 2 == 0
    assert len(pos) / 2 == len(nes) + 1
    chunks = [text[pos[i]: pos[i+1]] for i in range(0, len(pos), 2)]
    for i in range(len(chunks)):
        chunks[i] = chunks[i].replace('?', ' ?').replace('.', ' .')
    bingo = ''
    for i in range(len(chunks) - 1):
        bingo += chunks[i] + nes[i][0]
    bingo += chunks[-1]
    bingo.replace('  ?', ' ?').replace('  .', ' .')
    return bingo

