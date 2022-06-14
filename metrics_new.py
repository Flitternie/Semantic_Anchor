import os
import random
import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from utils.data import DataLoader
from utils.load_kb import DataForSPARQL
import logging
from metrics import evaluate, evaluate_kqapro, evaluate_overnight, evaluate_webnlg, evaluate_webnlg_reverse

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()
import warnings
warnings.simplefilter("ignore") 

def gather_and_extend(input, output):
    with torch.no_grad():
        all_x = [torch.zeros_like(input, device=input.device) for _ in range(dist.get_world_size())]
        dist.all_gather_multigpu(all_x, input)
        for x in all_x:
            output.extend(x.cpu().numpy())

def validate(args, model, data, device, tokenizer):
    if args.local_rank in [-1, 0]:
        logging.info("===================Dev==================")
    model.eval()
    all_outputs = []
    all_intermediate_outputs = []
    all_targets = []
    all_intermediate_targets = []
    all_answers = []
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            source_ids, source_mask, intermediate_target_ids, target_ids, answers = [x.to(device) for x in batch]

            outputs = model.module.generate(
                input_ids=source_ids,
                use_cache=True,
                max_length = args.eval_max_length,
                num_beams = args.beam_size,
                length_penalty=1.0,
                # return_dict_in_generate=True,
                # output_hidden_states=True,
            ) if hasattr(model, "module") else model.generate(
                input_ids=source_ids,
                use_cache=True,
                max_length = args.eval_max_length,
                num_beams = args.beam_size,
                length_penalty=1.0,
                # return_dict_in_generate=True,
                # output_hidden_states=True,
            )
            
            all_outputs.extend(outputs.cpu().numpy())
            all_targets.extend(target_ids.cpu().numpy())
            all_answers.extend(answers.cpu().numpy())
            # all_intermediate_outputs.extend(outputs[1].cpu().numpy())
            all_intermediate_targets.extend(intermediate_target_ids.cpu().numpy())
            
        assert len(all_outputs) == len(all_targets) 
        outputs = [tokenizer.decode(output_id, skip_special_tokens = True, clean_up_tokenization_spaces = False) for output_id in all_outputs]
        targets = [tokenizer.decode(target_id, skip_special_tokens = True, clean_up_tokenization_spaces = False) for target_id in all_targets]
        all_intermediate_targets = [tokenizer.decode(target_id, skip_special_tokens = True, clean_up_tokenization_spaces = False) for target_id in all_intermediate_targets]
        try:
            given_answer = [[data.vocab['answer_idx_to_token'][a] for a in al] for al in all_answers]
        except Exception as e:
            given_answer = None
        
        lf_matching, str_matching = evaluate(outputs, targets)
        if 'kqapro' in args.input_dir:
            lf_matching = evaluate_kqapro(args, given_answer, outputs)
        elif 'webnlg_reverse' in args.input_dir:
            lf_matching = evaluate_webnlg_reverse(args, outputs, targets)
        elif 'webnlg' in args.input_dir:
            lf_matching = evaluate_webnlg(args, outputs)
        elif 'overnight' in args.input_dir:
            lf_matching = evaluate_overnight(args, outputs, targets, all_answers)
        if args.local_rank in [-1, 0]:
            logging.info('Execution accuracy: {}, String matching accuracy: {}'.format(lf_matching, str_matching))

        return lf_matching, outputs
