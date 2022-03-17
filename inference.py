import os
import torch
import argparse
import sys

from utils.misc import MetricLogger, seed_everything, ProgressBar
from utils.load_kb import DataForSPARQL
from utils.data import DataLoader

from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer

import torch.optim as optim
import logging
import time
from utils.lr_scheduler import get_linear_schedule_with_warmup
import re

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()
import warnings
warnings.simplefilter("ignore") # hide warnings that caused by invalid sparql query

def inference(args):
    if args.mode == 'program':
        from metrics import validate

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.info("Create train_loader and val_loader.........")
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    val_pt = os.path.join(args.input_dir, 'test.pt')
    val_loader = DataLoader(vocab_json, val_pt, args.batch_size)
    
    logging.info("Create model.........")
    config_class, model_class, tokenizer_class = (BartConfig, BartForConditionalGeneration, BartTokenizer)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.ckpt)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)

    _, outputs = validate(args, model, val_loader, device, tokenizer)

            

def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True, help='path to save files')
    parser.add_argument('--model_name_or_path', required = True)
    parser.add_argument('--ckpt', required=True)

    # training parameters
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')

    parser.add_argument('--validate', action='store_false')

    # validating parameters
    # parser.add_argument('--num_return_sequences', default=1, type=int)

    # model hyperparameters
    parser.add_argument('--dim_hidden', default=1024, type=int)
    parser.add_argument('--alpha', default = 1e-4, type = float)

    args = parser.parse_args()

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

