import os
import torch
import argparse
import sys
import importlib.util

from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from utils.misc import seed_everything
from model import CustomizedBartForConditionalGeneration

import logging
import time
from metrics import validate

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()
import warnings
warnings.simplefilter("ignore") # hide warnings that caused by invalid sparql query

def inference(args):
    if args.customized:
        from utils.data_customized import DataLoader
    else:
        from utils.data import DataLoader

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.info("Create train_loader and test_loader.........")
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    test_pt = os.path.join(args.input_dir, 'test.pt')
    test_loader = DataLoader(vocab_json, test_pt, args.batch_size)
    
    logging.info("Create model.........")
    config_class, model_class, tokenizer_class = (AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer)
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

    acc, outputs = validate(args, model, test_loader, device, tokenizer)
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

    # training parameters
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument("--eval_max_length", default=500, type=int,
                        help="Eval max length.")
    parser.add_argument("--beam_size", default=1, type=int,
                        help="Beam size for inference.")

    # validating parameters
    # parser.add_argument('--num_return_sequences', default=1, type=int)

    # model hyperparameters
    parser.add_argument('--dim_hidden', default=1024, type=int)
    parser.add_argument('--alpha', default = 1e-4, type = float)

    parser.add_argument('--customized', action='store_true')
    parser.add_argument('--intermediate_layer', default=2, type=int)

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
        logging.info(k+':'+str(v))

    seed_everything(args.seed)

    inference(args)


if __name__ == '__main__':
    main()

