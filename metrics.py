import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from utils.load_kb import DataForSPARQL
import logging
import importlib.util


def evaluate_kqapro(args, given_answer, outputs):
    # evaluate on KQAPRO dataset
    kb = DataForSPARQL(os.path.join("./data/kqapro/data/", 'kb.json'))
    try:
        spec = importlib.util.spec_from_file_location("config", args.config)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
    except:
        raise Exception('Error loading config file')
    return config.evaluate(args, given_answer, outputs, kb)


def evaluate_webnlg(args, outputs):
    # evaluate on WebNLG dataset
    data_dir = './data/webnlg/data/'
    try:
        data_split = 'test_both' if args.inference else 'val'
    except:
        data_split = 'val'
    try:
        spec = importlib.util.spec_from_file_location("config", args.config)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
    except:
        raise Exception('Error loading config file')
    return config.evaluate(args, data_dir, outputs, data_split)


def evaluate_webnlg_reverse(args, outputs, targets):
    # evaluate on WebNLG-reverse dataset
    try:
        spec = importlib.util.spec_from_file_location("config", args.config)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
    except:
        raise Exception('Error loading config file')
    return config.evaluate(args, outputs, targets)


def evaluate_agenda(args, outputs, ):
    data_dir = './data/agenda/data/'
    try:
        data_split = 'test' if args.inference else 'val'
    except:
        data_split = 'val'
    try:
        spec = importlib.util.spec_from_file_location("config", args.config)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
    except:
        raise Exception('Error loading config file')
    return config.evaluate(args, data_dir, outputs, data_split)


def evaluate_overnight(args, outputs, targets, domains):
    try:
        spec = importlib.util.spec_from_file_location("config", args.config)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
    except:
        raise Exception('Error loading config file')
    return config.evaluate(args, outputs, targets, domains)


def evaluate_wikisql(args, outputs, targets):
    try:
        spec = importlib.util.spec_from_file_location("config", args.config)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
    except:
        raise Exception('Error loading config file')
    return config.evaluate(args, outputs, targets)


def evaluate(outputs, targets):
    assert len(outputs) == len(targets)
    return np.mean([1 if p.strip() == g.strip() else 0 for p, g in zip(outputs, targets)]), np.mean(
        [1 if p.strip().lower() == g.strip().lower() else 0 for p, g in zip(outputs, targets)])


def validate(args, model, data, device, tokenizer):
    if args.local_rank in [-1, 0]:
        logging.info("===================Dev==================")
    model.eval()
    all_outputs = []
    all_targets = []
    all_answers = []

    if args.customized:
        all_intermediate_outputs = []
        all_intermediate_targets = []
        logging.info(
            nn.functional.softmax(model.intermediate_weighting).cpu().tolist())  # weighting for intermediate layers

    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            if args.customized:
                if args.hybrid:
                    source_ids, source_mask, _, extra_intermediate_target_ids, extra_intermediate_target_mask, intermediate_target_ids, intermediate_target_mask, target_ids, answers = [
                        x.to(device) for x in batch]
                else:
                    source_ids, source_mask, _, intermediate_target_ids, intermediate_target_mask, target_ids, answers = [
                        x.to(device) for x in batch]
            else:
                source_ids, source_mask, _, target_ids, answers = [x.to(device) for x in batch]

            if args.customized:
                full_outputs = model.module.forward(
                    input_ids=source_ids,
                    use_cache=True,
                    return_dict=True
                ) if hasattr(model, "module") else model.forward(
                    input_ids=source_ids,
                    use_cache=True,
                    return_dict=True
                )
                intermediate_outputs = torch.argmax(full_outputs.intermediate_logits, dim=-1).cpu().numpy()
                for output_id in intermediate_outputs:
                    try:
                        eos_pos = output_id.tolist().index(tokenizer.eos_token_id)
                        output_id[eos_pos + 1:] = [tokenizer.pad_token_id] * (len(output_id) - eos_pos - 1)
                    except:
                        pass
                    all_intermediate_outputs.append(output_id)
                all_intermediate_targets.extend(intermediate_target_ids.cpu().numpy())

            outputs = model.module.generate(
                input_ids=source_ids,
                use_cache=True,
                max_length=args.eval_max_length,
                num_beams=args.beam_size,
                length_penalty=1.0,
            ) if hasattr(model, "module") else model.generate(
                input_ids=source_ids,
                use_cache=True,
                max_length=args.eval_max_length,
                num_beams=args.beam_size,
                length_penalty=1.0,
            )

            all_outputs.extend(outputs.cpu().numpy())
            all_targets.extend(target_ids.cpu().numpy())
            all_answers.extend(answers.cpu().numpy())

        assert len(all_outputs) == len(all_targets)

        if 'overnight' in args.input_dir:
            outputs = [tokenizer.decode(output_id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for
                       output_id in all_outputs]
            targets = [tokenizer.decode(target_id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for
                       target_id in all_targets]
        else:
            outputs = [tokenizer.decode(output_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for
                       output_id in all_outputs]
            targets = [tokenizer.decode(target_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for
                       target_id in all_targets]

        print("Target sample sequence: %s " % targets[-1])
        print("Output sample sequence: %s " % outputs[-1])
        if args.customized:
            all_intermediate_outputs = [
                tokenizer.decode(output_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output_id
                in all_intermediate_outputs]
            all_intermediate_targets = [
                tokenizer.decode(target_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for target_id
                in all_intermediate_targets]

        try:
            given_answer = [[data.vocab['answer_idx_to_token'][a] for a in [al]] for al in all_answers]
        except Exception as e:
            print(e)
            given_answer = None

    if args.customized:
        with open(os.path.join(args.output_dir, 'intermediate_layer_output.txt'), 'w') as f:
            for output in all_intermediate_outputs:
                f.write(output + '\n')
    with open(os.path.join(args.output_dir, 'output.txt'), 'w') as f:
        for output in outputs:
            f.write(output + '\n')

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
    elif 'wikisql' in args.input_dir:
        lf_matching = evaluate_wikisql(args, outputs, targets)
    if args.local_rank in [-1, 0]:
        logging.info('Execution accuracy: {}, String matching accuracy: {}'.format(lf_matching, str_matching))

    return lf_matching, outputs