import json
import pickle
import torch
from Bart.utils.misc import invert_dict

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, hybrid=False):
        self.hybrid = hybrid
        if hybrid:
            self.source_ids, self.source_mask, self.intermediate_target_ids, self.intermediate_target_mask, self.extra_intermediate_target_ids, self.extra_intermediate_target_mask, self.target_ids, self.extra_ids = inputs
        else:
            self.source_ids, self.source_mask, self.intermediate_target_ids, self.intermediate_target_mask, self.target_ids, self.extra_ids = inputs
        
    def __getitem__(self, index):
        source_ids = torch.LongTensor(self.source_ids[index])
        source_mask = torch.LongTensor(self.source_mask[index])
        intermediate_target_ids = torch.LongTensor(self.intermediate_target_ids[index])
        intermediate_target_mask = torch.LongTensor(self.intermediate_target_mask[index])
        if self.hybrid:
            extra_intermediate_target_ids = torch.LongTensor(self.extra_intermediate_target_ids[index])
            extra_intermediate_target_mask = torch.LongTensor(self.extra_intermediate_target_mask[index])
        target_ids = torch.LongTensor(self.target_ids[index])
        extra_ids = torch.LongTensor([self.extra_ids[index]])
        if self.hybrid:
            return source_ids, source_mask, intermediate_target_ids, intermediate_target_mask, extra_intermediate_target_ids, extra_intermediate_target_mask, target_ids, extra_ids
        else:
            return source_ids, source_mask, intermediate_target_ids, intermediate_target_mask, target_ids, extra_ids

    def __len__(self):
        return len(self.source_ids)


def load_vocab(path):
    try:
        vocab = json.load(open(path))
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
    except:
        vocab = None
    return vocab

def collate(batch):
    batch = list(zip(*batch))
    source_ids = torch.stack(batch[0])
    source_mask = torch.stack(batch[1])
    if batch[-1][0] is None:
        intermediate_target_ids, intermediate_target_mask, target_ids, extra_ids = None, None, None, None
    else:
        intermediate_target_ids = torch.stack(batch[2])
        intermediate_target_mask = torch.stack(batch[3])
        target_ids = torch.stack(batch[-2])
        extra_ids = torch.cat(batch[-1])
        if len(batch) > 6:
            extra_intermediate_target_ids = torch.stack(batch[4])
            extra_intermediate_target_mask = torch.stack(batch[5])
    if len(batch) > 6:
        return source_ids, source_mask, intermediate_target_ids, intermediate_target_mask, extra_intermediate_target_ids, extra_intermediate_target_mask, target_ids, extra_ids
    return source_ids, source_mask, intermediate_target_ids, intermediate_target_mask, target_ids, extra_ids

def prepare_dataset(vocab_json, question_pt, training=False, hybrid=False):
    vocab = load_vocab(vocab_json)
    inputs = []
    input_len = 6 if not hybrid else 8
    with open(question_pt, 'rb') as f:
        for _ in range(input_len):
            inputs.append(pickle.load(f))
    dataset = Dataset(inputs, hybrid=hybrid)
    return dataset, vocab


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, vocab_json, question_pt, batch_size, training=False, hybrid=False):
        dataset, vocab = prepare_dataset(vocab_json, question_pt, training, hybrid)
        super().__init__(dataset, batch_size=batch_size, shuffle=training, collate_fn=collate)
        self.vocab = vocab


class DistributedDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, vocab, batch_size, sampler):
        self.vocab = vocab
        self.sampler = sampler
        super().__init__(dataset, batch_size=batch_size, sampler=self.sampler, pin_memory=True, collate_fn=collate)