import os
import random
import torch
import numpy as np
from tqdm import tqdm
from utils.data import DataLoader
from utils.load_kb import DataForSPARQL
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()
import warnings
warnings.simplefilter("ignore") 

def evaluate_kqapro(args, given_answer, outputs):
    # evaluate on KQAPRO dataset
    kb = DataForSPARQL(os.path.join("./data/kqapro/data/", 'kb.json'))
    from data.kqapro.evaluate import evaluate 
    return evaluate(args, given_answer, outputs, kb)

def evaluate_webnlg(args, outputs):
    # evaluate on WebNLG dataset
    data_dir = './data/webnlg/data/'
    try:
        data_split = 'test_both' if args.inference else 'val'
    except:
        data_split = 'val'
    from data.webnlg.evaluate import evaluate
    return evaluate(args, data_dir, outputs, data_split)

def evaluate_webnlg_reverse(args, outputs, targets):
    # evaluate on WebNLG-reverse dataset
    from data.webnlg_reverse.evaluate import evaluate
    return evaluate(args, outputs, targets)

def evaluate_agenda(args, outputs,):
    data_dir = './data/agenda/data/'
    try:
        data_split = 'test' if args.inference else 'val'
    except:
        data_split = 'val'
    from data.agenda.evaluate import evaluate
    return evaluate(args, data_dir, outputs, data_split)

def evaluate_overnight(args, outputs, targets, domains):
    from data.overnight.evaluate import evaluate
    return evaluate(args, outputs, targets, domains)

def evaluate(outputs, targets):
    assert len(outputs) == len(targets)
    return np.mean([1 if p.strip() == g.strip() else 0 for p, g in zip(outputs, targets)]), np.mean([1 if p.strip().lower() == g.strip().lower() else 0 for p, g in zip(outputs, targets)])

def validate(args, model, data, device, tokenizer):
    if args.local_rank in [-1, 0]:
        logging.info("===================Dev==================")
    model.eval()
    all_outputs = []
    all_targets = []
    all_answers = []
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            source_ids, source_mask, _, target_ids, answers = [x.to(device) for x in batch]
            outputs = model.module.generate(
                input_ids=source_ids,
                use_cache=True,
                max_length = args.eval_max_length,
                num_beams = args.beam_size,
                length_penalty=1.0
            ) if hasattr(model, "module") else model.generate(
                input_ids=source_ids,
                use_cache=True,
                max_length = args.eval_max_length,
                num_beams = args.beam_size,
                length_penalty=1.0
            ) 

            all_outputs.extend(outputs.cpu().numpy())
            all_targets.extend(target_ids.cpu().numpy())
            all_answers.extend(answers.cpu().numpy())
            
        assert len(all_outputs) == len(all_targets) 
        outputs = [tokenizer.decode(output_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for output_id in all_outputs]
        targets = [tokenizer.decode(target_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for target_id in all_targets]
        try:
            given_answer = [data.vocab['answer_idx_to_token'][a] for a in all_answers]
        except:
            given_answer = None
            
    lf_matching, str_matching = evaluate(outputs, targets)
    if 'kqapro' in args.input_dir:
        lf_matching = evaluate_kqapro(args, given_answer, outputs)
    elif 'webnlg_reverse' in args.input_dir:
        lf_matching = evaluate_webnlg_reverse(args, outputs, targets)
    elif 'webnlg' in args.input_dir:
        lf_matching = evaluate_webnlg(args, outputs)
    elif 'agenda' in args.input_dir:
        lf_matching = evaluate_agenda(args, outputs)
    elif 'overnight' in args.input_dir:
        lf_matching = evaluate_overnight(args, outputs, targets, all_answers)
    if args.local_rank in [-1, 0]:
        logging.info('Execution accuracy: {}, String matching accuracy: {}'.format(lf_matching, str_matching))

    return lf_matching, outputs
