import json
import pickle
import torch
from utils.misc import invert_dict

def load_vocab(path):
    vocab = json.load(open(path))
    vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
    return vocab

def collate(batch):
    batch = list(zip(*batch))
    source_ids = torch.stack(batch[0])
    source_mask = torch.stack(batch[1])
    choices = torch.stack(batch[2])
    if batch[-1][0] is None:
        intermediate_target_ids, intermediate_target_mask, target_ids, answer = None, None, None, None
    else:
        if len(batch) > 7:
            extra_intermediate_target_ids = torch.stack(batch[-6])
            extra_intermediate_target_mask = torch.stack(batch[-5])
        intermediate_target_ids = torch.stack(batch[-4])
        intermediate_target_mask = torch.stack(batch[-3])
        target_ids = torch.stack(batch[-2])
        answer = torch.cat(batch[-1])
    if len(batch) > 7:
        return source_ids, source_mask, choices, extra_intermediate_target_ids, extra_intermediate_target_mask, intermediate_target_ids, intermediate_target_mask, target_ids, answer
    return source_ids, source_mask, choices, intermediate_target_ids, intermediate_target_mask, target_ids, answer

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, hybrid=False):
        self.hybrid = hybrid
        if hybrid:
            self.source_ids, self.source_mask, self.extra_intermediate_target_ids, self.extra_intermediate_target_mask, self.intermediate_target_ids, self.intermediate_target_mask, self.target_ids, self.choices, self.answers = inputs
        else:
            self.source_ids, self.source_mask, self.intermediate_target_ids, self.intermediate_target_mask, self.target_ids, self.choices, self.answers = inputs
        self.is_test = len(self.answers)==0
        
    def __getitem__(self, index):
        source_ids = torch.LongTensor(self.source_ids[index])
        source_mask = torch.LongTensor(self.source_mask[index])
        choices = torch.LongTensor(self.choices[index])
        intermediate_target_ids = torch.LongTensor(self.intermediate_target_ids[index])
        intermediate_target_mask = torch.LongTensor(self.intermediate_target_mask[index])
        if self.hybrid:
            extra_intermediate_target_ids = torch.LongTensor(self.extra_intermediate_target_ids[index])
            extra_intermediate_target_mask = torch.LongTensor(self.extra_intermediate_target_mask[index])
        if self.is_test:
            target_ids = None
            answer = None
        else:
            target_ids = torch.LongTensor(self.target_ids[index])
            answer = torch.LongTensor([self.answers[index]])
        if self.hybrid:
            return source_ids, source_mask, choices, extra_intermediate_target_ids, extra_intermediate_target_mask, intermediate_target_ids, intermediate_target_mask, target_ids, answer
        else:
            return source_ids, source_mask, choices, intermediate_target_ids, intermediate_target_mask, target_ids, answer

    def __len__(self):
        return len(self.source_ids)

class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, vocab_json, question_pt, batch_size, training=False, hybrid=False):
        vocab = load_vocab(vocab_json)
        
        inputs = []
        input_len = 7 if not hybrid else 9
        with open(question_pt, 'rb') as f:
            for _ in range(input_len):
                inputs.append(pickle.load(f))
        dataset = Dataset(inputs, hybrid=hybrid)
        super().__init__(
            dataset, 
            batch_size=batch_size,
            shuffle=training,
            collate_fn=collate, 
            )
        self.vocab = vocab

def prepare_dataset(vocab_json, question_pt, training=False, hybrid=False):
    vocab = load_vocab(vocab_json)
    
    inputs = []
    input_len = 7 if not hybrid else 9
    with open(question_pt, 'rb') as f:
        for _ in range(input_len):
            inputs.append(pickle.load(f))
    dataset = Dataset(inputs, hybrid=hybrid)
    return dataset, vocab

class DistributedDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, vocab, batch_size, sampler):
        self.vocab = vocab
        self.sampler = sampler
        super().__init__(
            dataset, 
            batch_size=batch_size,
            sampler=self.sampler,
            pin_memory=True,
            collate_fn=collate, 
            )