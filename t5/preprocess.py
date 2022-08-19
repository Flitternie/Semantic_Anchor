import os
import re
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import importlib.util

from transformers import AutoTokenizer, set_seed


def get_key_info(ir: str):
    key_info = re.findall(r"(?<=\<[A-Z]\>)[^\<\>]*(?=\<\/[A-Z]\>)", ir)
    key_info = list(dict.fromkeys(key_info).keys()) # set operation to remove duplicates but keep order
    key_info = [item.strip() for item in key_info]
    key_info += [item.replace(" ", "_") for item in key_info if item.islower()]
    return key_info + [" {}".format(item) for item in key_info] + ["{} ".format(item) for item in key_info]


def extract_first_order_syntax(ir):
    key_info = re.findall(r"(?<=\<[A-Z]\>)[^\<\>]*(?=\<\/[A-Z]\>)", ir)
    key_info = list(dict.fromkeys(key_info).keys()) # set operation to remove duplicates but keep order
    return key_info


def encode_dataset(args, dataset, tokenizer):
    
    def get_mask_ids(info_ids: list, target_ids: list):
        checked_ids = []
        temp_ids = []
        j = 0
        for i in range(len(target_ids)):
            if target_ids[i] == tokenizer.pad_token_id:
                break
            if j == len(info_ids):
                checked_ids.extend(temp_ids)
                j, temp_ids = 0, []
            if target_ids[i] == info_ids[j]:
                temp_ids.append(i)
                j += 1
            else:
                j, temp_ids = 0, []
                if target_ids[i] == info_ids[j]:
                    temp_ids.append(i)
                    j += 1
        return checked_ids
    
    def prepare_long_supervision(intermediate_targets, intermediate_key_info):
        intermediate_targets = tokenizer.batch_encode_plus(intermediate_targets, max_length = max_seq_length, padding='max_length', truncation = True)
        intermediate_target_ids = np.array(intermediate_targets['input_ids'], dtype=np.int32)
        intermediate_target_mask = np.zeros_like(intermediate_target_ids, dtype=np.int32)
        
        print("Masking tokens...")
        for i in tqdm(range(len(intermediate_key_info))):
            for j in range(len(intermediate_key_info[i])):
                key_ids = tokenizer.encode(intermediate_key_info[i][j])[:-1]
                mask_ids = get_mask_ids(key_ids, intermediate_target_ids[i])
                np.put(intermediate_target_mask[i], mask_ids, 1)
                eos_pos = intermediate_target_ids[i].tolist().index(tokenizer.eos_token_id)
                intermediate_target_mask[i][0] = 1
                intermediate_target_mask[i][eos_pos] = 1
        return intermediate_targets, intermediate_target_ids, intermediate_target_mask

    inputs = []
    targets = []
    extra_ids = []
    if args.customized:
        if args.supervision_form == "long":
            intermediate_targets = []
            intermediate_key_info = []
        elif args.supervision_form == "short":
            intermediate_targets = []
        elif args.supervision_form == "hybrid":
            intermediate_short_targets = []
            intermediate_long_targets = []
            intermediate_key_info = []

    for item in tqdm(dataset):
        if item['input'] is None or item['target'] is None:
            continue
        inputs.append(item['input'])
        targets.append(item['target'])
        if 'extra_id' in item.keys():
            extra_ids.append(item['extra_id'])
        if args.customized:
            if args.supervision_form == "long":
                intermediate_targets.append(item["target"])
                intermediate_key_info.append(get_key_info(item['ir']))
            elif args.supervision_form == "short":
                intermediate_targets.append(" ".join(extract_first_order_syntax(item['ir'])).strip())
            elif args.supervision_form == "hybrid":
                intermediate_short_targets.append(" ".join(extract_first_order_syntax(item['ir'])).strip())
                intermediate_long_targets.append(item["target"])
                intermediate_key_info.append(get_key_info(item['ir']))
    
    assert len(inputs) == len(targets) 

    sequences = inputs + targets
    encoded_inputs = tokenizer.batch_encode_plus(sequences, padding='longest')
    max_seq_length = min(len(encoded_inputs['input_ids'][0]), args.max_length)

    input_ids = tokenizer.batch_encode_plus(inputs, max_length=max_seq_length, padding='max_length', truncation=True)
    source_ids = np.array(input_ids['input_ids'], dtype=np.int32)
    source_mask = np.array(input_ids['attention_mask'], dtype=np.int32)
    
    target_ids = tokenizer.batch_encode_plus(targets, max_length=max_seq_length, padding = 'max_length', truncation = True)
    target_ids = np.array(target_ids['input_ids'], dtype = np.int32)

    if args.customized:        
        if args.supervision_form == "short":
            intermediate_targets = tokenizer.batch_encode_plus(intermediate_targets, max_length = max_seq_length, padding='max_length', truncation = True)
            intermediate_target_ids = np.array(intermediate_targets['input_ids'], dtype=np.int32)
            intermediate_target_mask = np.array(intermediate_targets['attention_mask'], dtype=np.int32)
        
        elif args.supervision_form == "long":
            intermediate_targets, intermediate_target_ids, intermediate_target_mask = prepare_long_supervision(intermediate_targets, intermediate_key_info)
            
        elif args.supervision_form == "hybrid":
            intermediate_short_targets = tokenizer.batch_encode_plus(intermediate_short_targets, max_length = max_seq_length, padding='max_length', truncation = True)
            intermediate_short_target_ids = np.array(intermediate_short_targets['input_ids'], dtype=np.int32)
            intermediate_short_target_mask = np.array(intermediate_short_targets['attention_mask'], dtype=np.int32)
            intermediate_long_targets, intermediate_long_target_ids, intermediate_long_target_mask = prepare_long_supervision(intermediate_long_targets, intermediate_key_info)

    extra_ids = np.array(extra_ids) if extra_ids else np.array([0]*len(inputs), dtype=np.int32)
    
    if args.customized:
        if args.supervision_form == "long" or args.supervision_form == "short":
            return source_ids, source_mask, intermediate_target_ids, intermediate_target_mask, target_ids, extra_ids
        elif args.supervision_form == "hybrid":
            return source_ids, source_mask, intermediate_short_target_ids, intermediate_short_target_mask, intermediate_long_target_ids, intermediate_long_target_mask, target_ids, extra_ids
    else:
       return source_ids, source_mask, target_ids, extra_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--max_length', default=512, type=int)

    parser.add_argument('--customized', action='store_true')
    parser.add_argument('--supervision_form', choices=['long', 'short', 'hybrid'])
    args = parser.parse_args()
    set_seed(42)

    if bool(args.customized) ^ bool(args.supervision_form):
        raise ValueError("args.customized and args.supervision_form must be co-specified.")
    
    for k, v in vars(args).items():
        print(k+':'+str(v))

    try:
        spec = importlib.util.spec_from_file_location("config", args.config)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        train_set, val_set, test_set, *xargs = config.load_data(args)
        task_special_tokens = config.special_tokens
    except:
        raise Exception('Error loading config file')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    if xargs:
        fn = os.path.join(args.output_dir, 'vocab.json')
        with open(fn, 'w') as f:
            json.dump(xargs[0], f, indent=2)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_tokens(task_special_tokens)
    print('Tokenizer loaded with domain specific special tokens added:')
    print(tokenizer.get_added_vocab())
    
    for name, dataset in zip(('train', 'val', 'test'), (train_set, val_set, test_set)):
        print('Encode {} set'.format(name))
        outputs = encode_dataset(args, dataset, tokenizer)
        with open(os.path.join(args.output_dir, '{}.pt'.format(name)), 'wb') as f:
            for o in outputs:
                pickle.dump(o, f)


if __name__ == '__main__':
    main()