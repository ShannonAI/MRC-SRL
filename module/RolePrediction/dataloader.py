import os
import math
import json
import shutil

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler


LABELS = None
LABELS2ID = None


def batch_by_tokens(length, max_tokens):
    indexes = []
    i = 0
    while i < len(length):
        for j in range(i, len(length)):
            maxc = max(length[i:j+1])
            maxn = max(maxc, length[min(j+1, len(length)-1)])
            current_batch_tokens = maxc*(j+1-i)
            next_batch_tokens = maxn*(j+2-i)
            if (current_batch_tokens <= max_tokens and next_batch_tokens > max_tokens) or j == len(length)-1:
                indexes.append((i, j))
                i = j+1
                break
    return indexes


class MyDataset:
    def __init__(self, path="", tokenizer=None, max_tokens=1024):
        if not path:
            return
        data = json.load(open(path))
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target = []
        self.init_data(data, tokenizer, max_tokens)

    def init_data(self, data, tokenizer, max_tokens):
        #get initial features
        for d in tqdm(data, desc='preprocessing'):
            sentence = d['sentence']
            if 'roberta' in tokenizer.name_or_path:
                for i in range(1, len(sentence)):
                    sentence[i] = ' '+sentence[i]
            arguments = d['arguments']
            predicates = d['predicates']
            sentence1 = [tokenizer.tokenize(w) for w in sentence]
            for i in range(len(predicates)):
                pre = predicates[i]
                args = arguments[i]
                sentence1i = sentence1[:pre]+[['<p>']] + \
                    sentence1[pre:pre+1]+[['</p>']]+sentence1[pre+1:]
                if 'roberta' in tokenizer.name_or_path:
                    sentence2i = ['<s>']+sum(sentence1i, [])+['</s>']
                else:
                    sentence2i = ['[CLS]']+sum(sentence1i, [])+['[SEP]']
                input_ids = tokenizer.convert_tokens_to_ids(sentence2i)
                self.input_ids.append(torch.tensor(input_ids, dtype=torch.long))
                args1 = set() #role set
                for arg in args:
                    s, e, l = arg
                    l = l.split('-')
                    key = l[-1]
                    if key in LABELS:
                        args1.add(key)
                    else:
                        pass
                target = torch.zeros(len(LABELS), dtype=torch.uint8)
                for key in args1:
                    j = LABELS2ID[key]
                    target[j] = 1
                self.target.append(target)
        self.do_batch(max_tokens)

    def do_batch(self, max_tokens):
        self.batch_input_ids = []
        self.batch_target = []
        self.batch_attention_mask = []
        t = zip(self.input_ids, self.target)
        #sort by length
        self.input_ids, self.target = zip(*sorted(t, key=lambda x: len(x[0])))
        length = [len(c) for c in self.input_ids]
        length = np.array(length)
        #process input that exceeds max tokens 
        length[length > max_tokens] = max_tokens
        indexes = batch_by_tokens(length, max_tokens)
        #batch by tokens
        for s, e in tqdm(indexes, desc='batching'):
            input_ids = self.input_ids[s:e+1]
            target = self.target[s:e+1]
            input_ids1 = pad_sequence(input_ids, batch_first=True)
            attention_mask = torch.zeros(input_ids1.shape)
            for i in range(len(input_ids1)):
                attention_mask[i, :len(input_ids[i])] = 1
            target1 = torch.stack(target, dim=0)
            self.batch_input_ids.append(input_ids1)
            self.batch_target.append(target1)
            self.batch_attention_mask.append(attention_mask)

    def save(self, save_dir):
        #save processed data
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
            os.makedirs(save_dir)
        else:
            os.makedirs(save_dir)
        input_ids = [d.numpy() for d in self.input_ids]
        target = [d.numpy() for d in self.target]
        np.save(os.path.join(save_dir, 'input_ids.npy'), input_ids)
        np.save(os.path.join(save_dir, 'target.npy'), target)

    def load(self, save_dir, max_tokens):
        #load cached data
        input_ids = np.load(os.path.join(
            save_dir, 'input_ids.npy'), allow_pickle=True)
        target = np.load(os.path.join(
            save_dir, 'target.npy'), allow_pickle=True)
        self.input_ids = [torch.from_numpy(d) for d in input_ids]
        self.target = [torch.from_numpy(d) for d in target]
        self.do_batch(max_tokens)

    def __getitem__(self, i):
        return {'input_ids': self.batch_input_ids[i].long(), 'attention_mask': self.batch_attention_mask[i].float(), 'target': self.batch_target[i].float()}

    def __len__(self):
        return len(self.batch_input_ids)


def load_data(path, pretrained_model_name_or_path, max_tokens, shuffle, dataset_tag, local_rank=-1):
    if dataset_tag == 'conll2005' or dataset_tag == 'conll2009':
        ARGS = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'AA']
        ARGMS = ['DIR', 'LOC', 'MNR', 'TMP', 'EXT', 'REC',
                 'PRD', 'PNC', 'CAU', 'DIS', 'ADV', 'MOD', 'NEG']
    elif dataset_tag == 'conll2012':
        ARGS = ['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5', 'ARGA']
        ARGMS = ['MNR', 'ADV', 'LOC', 'TMP', 'PRP', 'PRD', 'DIR', 'DIS', 'MOD',
                 'NEG', 'CAU', 'EXT', 'LVB', 'REC', 'ADJ', 'GOL', 'DSP', 'PRR', 'COM', 'PRX', 'PNC']
    global LABELS
    global LABELS2ID
    LABELS = ARGS+ARGMS
    LABELS2ID = {k: v for v, k in enumerate(LABELS)}
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<p>', '</p>']})
    dataset = MyDataset(path, tokenizer, max_tokens)
    sampler = DistributedSampler(
        dataset, rank=local_rank) if local_rank != -1 else None
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler,
                            shuffle=shuffle if sampler is None else False, collate_fn=lambda x: x[0])
    return dataloader


def reload_data(path, max_tokens, shuffle=False, local_rank=-1):
    dataset = MyDataset()
    dataset.load(path, max_tokens)
    sampler = DistributedSampler(
        dataset, rank=local_rank) if local_rank != -1 else None
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler,
                            shuffle=shuffle if sampler is None else False, collate_fn=lambda x: x[0])
    return dataloader
