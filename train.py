import os
import sys
import time
import logging
import argparse
import importlib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
import transformers.utils.logging as transformers_logging 
from model import CustomizedBartForConditionalGeneration

from utils.misc import seed_everything, ProgressBar
from utils.lr_scheduler import get_linear_schedule_with_warmup

from inference import validate

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()
import warnings
warnings.simplefilter("ignore") 

def train(args):
    if args.customized:
        from utils.data_customized import DataLoader, DistributedDataLoader, prepare_dataset
    else:
        from utils.data import DataLoader, DistributedDataLoader, prepare_dataset
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.local_rank in [-1, 0]:
        logging.info("Create train_loader and val_loader.........")
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    train_pt = os.path.join(args.input_dir, 'train.pt')
    val_pt = os.path.join(args.input_dir, 'val.pt')
    
    if args.n_gpus > 1:
        train_dataset, train_vocab = prepare_dataset(vocab_json, train_pt, training=True, hybrid=args.hybrid)
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DistributedDataLoader(train_dataset, train_vocab, args.batch_size//args.n_gpus, train_sampler)
    else:
        train_loader = DataLoader(vocab_json, train_pt, args.batch_size, training=True, hybrid=args.hybrid)
    val_loader = DataLoader(vocab_json, val_pt, args.batch_size//args.n_gpus*2, training=False, hybrid=args.hybrid)
    
    if args.local_rank in [-1, 0]:
        logging.info("Create model.........")
    if args.customized:
        _, model_class, tokenizer_class = (AutoConfig, CustomizedBartForConditionalGeneration, AutoTokenizer)
    else:
        _, model_class, tokenizer_class = (AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    try:
        spec = importlib.util.spec_from_file_location("config", args.config)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        task_special_tokens = config.special_tokens
        tokenizer.add_tokens(task_special_tokens)
        if args.local_rank in [-1, 0]:
            logging.info("Add {} special tokens.".format(len(task_special_tokens)))
    except:
        raise Exception('Error loading config file')

    transformers_logging.set_verbosity_error()
    if args.local_rank in [-1, 0]:
        logging.info("Initiating model parameters.........")
    model = model_class.from_pretrained(args.ckpt) if args.ckpt else model_class.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    
    if args.n_gpus > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    else:
        model = model.to(device)

    t_total = len(train_loader) // args.gradient_accumulation_steps * args.num_train_epochs    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    bart_param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bart_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in bart_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate}
    ]
    args.warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Train!
    if args.local_rank in [-1, 0]:
        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(train_loader.dataset))
        logging.info("  Num Epochs = %d", args.num_train_epochs)
        logging.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logging.info("  Total optimization steps = %d", t_total)

    # Check if continuing training from a checkpoint
    if args.ckpt and args.local_rank in [-1, 0]:
        logging.info("Continuing training from checkpoint, will skip to saved global_step")
    
    global_step = 0
    tr_loss = 0.0
    best_acc, current_acc = 0.0, 0.0
    alpha = 1

    model.zero_grad()
    if args.local_rank in [-1, 0]:
        # current_acc, _ = validate(args, model, val_loader, device, tokenizer)
        print("Current performance on validation set: %f" % (current_acc))
    
    epochs_not_improving = 0

    for epoch_i in range(int(args.num_train_epochs)):
        if args.n_gpus > 1:
            train_loader.sampler.set_epoch(epoch_i)
        pbar = ProgressBar(n_total=len(train_loader), desc='Training')
        if args.local_rank in [-1, 0]:
            epochs_not_improving += 1
        for step, batch in enumerate(train_loader):

            model.train()
            batch = tuple(t.to(device) for t in batch)
            pad_token_id = tokenizer.pad_token_id

            if args.customized:
                if args.hybrid:
                    assert len(batch) == 8
                    source_ids, source_mask, extra_intermediate_ids, extra_intermediate_mask, intermediate_ids, intermediate_mask, y = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]
                    extra_intermediate_labels = extra_intermediate_ids[:, 1:].clone()
                    extra_intermediate_labels[extra_intermediate_ids[:, 1:] == pad_token_id] = -100
                    extra_intermediate_masks = extra_intermediate_mask[:, 1:].clone()
                else:
                    source_ids, source_mask, intermediate_ids, intermediate_mask, y = batch[0], batch[1], batch[2], batch[3], batch[4]
                intermediate_labels = intermediate_ids[:, 1:].clone()
                intermediate_labels[intermediate_ids[:, 1:] == pad_token_id] = -100
                intermediate_masks = intermediate_mask[:, 1:].clone()

            else:
                source_ids, source_mask, y = batch[0], batch[1], batch[-2]
            y_ids = y[:, :-1].contiguous()
            labels = y[:, 1:].clone()
            labels[y[:, 1:] == pad_token_id] = -100

            inputs = {
                "input_ids": source_ids.to(device),
                "attention_mask": source_mask.to(device),
                "decoder_input_ids": y_ids.to(device),
                "labels": labels.to(device),
            }
            if args.customized:
                inputs["intermediate_labels"] = intermediate_labels.to(device)
                inputs["intermediate_masks"] = intermediate_masks.to(device)
                inputs["alpha"] = alpha
                if args.hybrid:
                    inputs["extra_intermediate_labels"] = extra_intermediate_labels.to(device)
                    inputs["extra_intermediate_masks"] = extra_intermediate_masks.to(device)

            outputs = model(**inputs)
            loss = outputs[0]
            if args.n_gpus > 1:
                loss = loss.sum()

            loss.backward()
            loss_num = loss.item()
            pbar(step, {'loss': loss_num})
            tr_loss += loss_num

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
        
        if args.n_gpus > 1:
            dist.barrier()
        
        if args.local_rank in [-1, 0]:
            logging.info("Epoch %d loss: %.3f" % (epoch_i, loss_num))
            current_acc, _ = validate(args, model, val_loader, device, tokenizer)
            print("Current best performance on validation set: %f" % (current_acc))
        
        if args.local_rank in [-1, 0] and current_acc > best_acc:
            epochs_not_improving = 0
            best_acc = current_acc
            print("Best performance on validation set updated: %f" % (best_acc))
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, "checkpoint-best")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            # Take care of distributed/parallel training
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  
            model_to_save.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logging.info("Saving model checkpoint to %s", output_dir)
            tokenizer.save_vocabulary(output_dir)
            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            logging.info("Saving optimizer and scheduler states to %s", output_dir)
            
        if args.customized:
            if 'cuda' in str(device):
                torch.cuda.empty_cache()
            
            deltas = []
            print("Adjusting intermediate weights...")
            model.train()
            sampled_data = iter(train_loader)
            for i in range(args.sample_number):
                ad_batch = next(sampled_data)
                model.zero_grad()
                ad_batch = tuple(t.to(device) for t in ad_batch)
                pad_token_id = tokenizer.pad_token_id
                source_ids, source_mask, intermediate, intermediate_mask, y = ad_batch[0], ad_batch[1], ad_batch[2], ad_batch[3], ad_batch[4]
                intermediate_labels = intermediate[:, 1:].clone()
                intermediate_labels[intermediate[:, 1:] == pad_token_id] = -100
                intermediate_masks = intermediate_mask[:, 1:].clone()
                y_ids = y[:, :-1].contiguous()
                labels = y[:, 1:].clone()
                labels[y[:, 1:] == pad_token_id] = -100
                inputs = {
                    "input_ids": source_ids.to(device),
                    "attention_mask": source_mask.to(device),
                    "decoder_input_ids": y_ids.to(device),
                    "intermediate_labels": intermediate_labels.to(device),
                    "intermediate_masks": intermediate_masks.to(device),
                    "labels": labels.to(device),
                    "alpha": alpha,
                }
                outputs = model(**inputs)
                target_loss, intermediate_loss = outputs[-2], outputs[-1]
                if args.n_gpus > 1:
                    target_loss = target_loss.sum()
                    intermediate_loss = intermediate_loss.sum()

                intermediate_loss = alpha * intermediate_loss + 0 * target_loss  # to avoid unused param error
                intermediate_loss.backward()
                optimizer.step()
                model.zero_grad()
                outputs = model(**inputs)
                updated_target_loss = outputs[-2]
                if args.n_gpus > 1:
                    updated_target_loss = updated_target_loss.sum()
                delta = target_loss.item() / updated_target_loss.item()
                deltas.append(delta)
                model.zero_grad()

            assert len(deltas) == args.sample_number
            deltas = torch.tensor(deltas).to(device)
            s = torch.cuda.Stream()
            if args.n_gpus > 1:
                handle = dist.all_reduce(deltas, async_op=True)
                handle.wait()
                with torch.cuda.stream(s):
                    s.wait_stream(torch.cuda.default_stream())
                deltas = deltas / args.n_gpus

            alpha = min(alpha * ((sum(deltas) / len(deltas))**args.adjustment_step), args.max_alpha)
            if args.local_rank in [-1, 0]:
                logging.info("The updated alpha is: %f" % alpha)
        
        if args.n_gpus > 1:
            dist.barrier()

        if 'cuda' in str(device):
            torch.cuda.empty_cache()
        if epochs_not_improving > args.early_stopping:
            logging.info("%d epochs not improving, training early stopped" % epochs_not_improving)
            dist.destroy_process_group()
            return global_step, tr_loss / global_step
        
    return global_step, tr_loss / global_step


def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--ckpt', default=None)

    # training parameters
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--learning_rate', default=3e-5, type=float)
    parser.add_argument('--num_train_epochs', default=25, type=int)
    parser.add_argument('--early_stopping', default=5, type=int)
    parser.add_argument('--warmup_proportion', default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1=10% of training.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    
    parser.add_argument("--eval_max_length", default=500, type=int,
                        help="Eval max length.")
    parser.add_argument("--beam_size", default=1, type=int,
                        help="Beam size for inference.")

    # special parameters
    parser.add_argument('--customized', action='store_true')
    parser.add_argument('--hybrid', action='store_true')
    parser.add_argument('--adjustment_step', default=2, type=float)
    parser.add_argument("--sample_number", default=10, type=int)
    parser.add_argument("--max_alpha", default=1.5, type=float)

    parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
    parser.add_argument('--port', default=12355, type=int)
    
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    fileHandler = logging.FileHandler(os.path.join(args.output_dir, '{}.log'.format(time_)))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    # args display
    if args.local_rank in [-1, 0]:
        for k, v in vars(args).items():
            logging.info(k+':'+str(v))

    seed_everything(args.seed)

    # distributed data parallel   
    args.n_gpus = torch.cuda.device_count()
    if args.n_gpus > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(args.port)
        dist.init_process_group(backend='nccl', world_size=args.n_gpus)
        torch.cuda.set_device(args.local_rank)

    train(args)
    
    if args.n_gpus > 1:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
