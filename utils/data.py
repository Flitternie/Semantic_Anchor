import json
import pickle
import torch
from utils.misc import invert_dict

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs):
        self.source_ids, self.source_mask, self.target_ids, self.extra_ids = inputs

    def __getitem__(self, index):
        source_ids = torch.LongTensor(self.source_ids[index])
        source_mask = torch.LongTensor(self.source_mask[index])
        target_ids = torch.LongTensor(self.target_ids[index])
        extra_ids = torch.LongTensor([self.extra_ids[index]])
        return source_ids, source_mask, target_ids, extra_ids

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
        target_ids, extra_ids = None, None
    else:
        target_ids = torch.stack(batch[2])
        extra_ids = torch.cat(batch[3])
    return source_ids, source_mask, target_ids, extra_ids

def prepare_dataset(vocab_json, question_pt, training=False, **kwargs):
    vocab = load_vocab(vocab_json)
    inputs = []
    input_len = 4
    with open(question_pt, 'rb') as f:
        for _ in range(input_len):
            inputs.append(pickle.load(f))
    dataset = Dataset(inputs)
    return dataset, vocab


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, vocab_json, question_pt, batch_size, training=False, **kwargs):
        dataset, vocab = prepare_dataset(vocab_json, question_pt, training)
        super().__init__(dataset, batch_size=batch_size, shuffle=training, collate_fn=collate)
        self.vocab = vocab


class DistributedDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, vocab, batch_size, sampler, **kwargs):
        self.vocab = vocab
        self.sampler = sampler
        super().__init__(dataset, batch_size=batch_size, sampler=self.sampler, pin_memory=True, collate_fn=collate)