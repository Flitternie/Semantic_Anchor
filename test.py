import json

import torch
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss

from dataclasses import dataclass
from typing import Optional, Tuple

from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput, Seq2SeqLMOutput
from transformers.models.bart.modeling_bart import BartModel, BartPretrainedModel, shift_tokens_right
from transformers.models.bart.configuration_bart import BartConfig
from transformers import BartTokenizerFast
from transformers.file_utils import ModelOutput
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoConfig, AutoTokenizer
import transformers.utils.logging as transformers_logging
from model import CustomizedBartForConditionalGeneration

from utils.misc import MetricLogger, seed_everything, ProgressBar
from utils.data import DataLoader, DistributedDataLoader, prepare_dataset
from utils.lr_scheduler import get_linear_schedule_with_warmup
import os
import re


def extract_first_order_syntax(ir: str) -> [str]:
    return " ".join(re.findall(r"(?<=\<[A-Z]\>)[^\<\>]*(?=\<\/[A-Z]\>)", ir))


if __name__ == "__main__":
    input_dir = "../../../../ldata/sjd/UIR/kqapro"
    vocab_json = os.path.join(input_dir, 'vocab.json')
    train_pt = os.path.join(input_dir, 'train.pt')
    val_pt = os.path.join(input_dir, 'val.pt')
    tok = AutoTokenizer.from_pretrained('../../../../ldata/sjd/bart-base/')
    train_dataset, train_vocab = prepare_dataset(vocab_json, train_pt, training=True)
    print(tok.decode(train_dataset[0][3]))
    print(tok.decode(train_dataset[0][2]))
