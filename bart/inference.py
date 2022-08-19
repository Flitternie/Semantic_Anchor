import os
import sys

import torch
import torch.nn as nn
import numpy as np

import argparse
import importlib.util
from tqdm import tqdm

from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from utils.misc import seed_everything
from bart.model import CustomizedBartForConditionalGeneration

import logging
import time

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()


def validate(args, model, data, device, tokenizer):
    try:
        spec = importlib.util.spec_from_file_location("config", args.config)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
    except:
        raise Exception('Error loading config file')

    args.tokenizer = tokenizer
    model.eval()
    model = model.module if hasattr(model, "module") else model

    all_outputs = []
    all_targets = []
    all_extra_ids = []

    if args.customized:
        all_intermediate_outputs = []
        all_intermediate_targets = []
        sublayer_outputs = [[] for _ in range(model.num_hybrid_layers)]
        logging.info(nn.functional.softmax(model.intermediate_weighting, dim=0).cpu().tolist())  # weighting for intermediate layers
        if args.hybrid:
            all_extra_intermediate_outputs = []
            all_extra_intermediate_targets = []
            logging.info(nn.functional.softmax(model.extra_intermediate_weighting, dim=0).cpu().tolist())  # weighting for intermediate layers

    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            if args.customized:
                if args.hybrid:
                    source_ids, _, extra_intermediate_target_ids, _, intermediate_target_ids, _, target_ids, extra_ids = [x.to(device) for x in batch]
                else:
                    source_ids, _, intermediate_target_ids, _, target_ids, extra_ids = [x.to(device) for x in batch]
            else:
                source_ids, _, target_ids, extra_ids = [x.to(device) for x in batch]

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
                
                for i in range(model.num_hybrid_layers):
                    sublayer_logits = model.lm_head(full_outputs.decoder_hidden_states[i+1]) + model.final_logits_bias
                    sublayer_outputs[i].extend(torch.argmax(sublayer_logits, dim=-1).cpu().numpy())
                all_intermediate_outputs.extend(intermediate_outputs)
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
            all_extra_ids.extend(extra_ids.cpu().numpy())

        assert len(all_outputs) == len(all_targets)
        outputs = tokenizer.batch_decode(all_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        targets = tokenizer.batch_decode(all_targets, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        print("Target sample sequence: %s " % targets[-1])
        print("Output sample sequence: %s " % outputs[-1])

        if args.customized:
            all_intermediate_outputs = tokenizer.batch_decode(all_intermediate_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if args.hybrid:
                all_extra_intermediate_outputs = tokenizer.batch_decode(all_extra_intermediate_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    with open(os.path.join(args.output_dir, 'output.txt'), 'w') as f:
        for output in outputs:
            f.write(output + '\n')

    if args.customized:
        with open(os.path.join(args.output_dir, 'intermediate_output.txt'), 'w') as f:
            for output in all_intermediate_outputs:
                f.write(output + '\n')
        for i in range(model.num_hybrid_layers):
            sublayer_output = [tokenizer.decode(output_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for output_id in sublayer_outputs[i]]
            with open(os.path.join(args.output_dir, 'layer_%d_output.txt' % (i+1)), 'w') as f:
                for output in sublayer_output:
                    f.write(output + '\n')
        if args.hybrid:
            with open(os.path.join(args.output_dir, 'extra_intermediate_output.txt'), 'w') as f:
                for output in all_extra_intermediate_outputs:
                    f.write(output + '\n')

    str_matching = np.mean([1 if p.strip() == g.strip() else 0 for p, g in zip(outputs, targets)])
    lf_matching = config.evaluate(args, outputs, targets, all_extra_ids, data)
    logging.info('Execution accuracy: {}, String matching accuracy: {}'.format(lf_matching, str_matching))

    return lf_matching, outputs


def inference(args):
    if args.customized:
        from utils.data_customized import DataLoader
    else:
        from utils.data import DataLoader

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.info("Create train_loader and test_loader.........")
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    test_pt = os.path.join(args.input_dir, 'test.pt')
    test_loader = DataLoader(vocab_json, test_pt, args.batch_size, training=False, hybrid=args.hybrid)

    logging.info("Create model.........")
    config_class, model_class, tokenizer_class = AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    try:
        spec = importlib.util.spec_from_file_location("config", args.config)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        task_special_tokens = config.special_tokens
        tokenizer.add_tokens(task_special_tokens)
    except:
        raise Exception('Error loading config file')
    if args.customized:
        model = CustomizedBartForConditionalGeneration.from_pretrained(args.ckpt)
    else:
        model = model_class.from_pretrained(args.ckpt)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    model.hybrid = True if args.hybrid else False

    _, outputs = validate(args, model, test_loader, device, tokenizer)
    with open("output.txt", "w") as f:
        for output in outputs:
            f.write(output + "\n")


def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--config', required=True)

    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--ckpt', required=True)

    # inference parameters
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument("--eval_max_length", default=500, type=int,
                        help="Eval max length.")
    parser.add_argument("--beam_size", default=1, type=int,
                        help="Beam size for inference.")

    # parser.add_argument('--num_return_sequences', default=1, type=int)

    # model hyperparameters
    parser.add_argument('--dim_hidden', default=1024, type=int)
    parser.add_argument('--alpha', default=1e-4, type=float)

    parser.add_argument('--customized', action='store_true')
    parser.add_argument('--hybrid', action='store_true')

    args = parser.parse_args()
    args.inference = True
    args.local_rank = -1

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    fileHandler = logging.FileHandler(os.path.join(args.output_dir, '{}.predict.log'.format(time_)))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    for k, v in vars(args).items():
        logging.info(k + ':' + str(v))

    seed_everything(args.seed)

    inference(args)


if __name__ == '__main__':
    main()
