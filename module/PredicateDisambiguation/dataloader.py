import os
from pdb import set_trace
import shutil
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler


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


frames = None
lemma_dict = {}


class MyDataset:
    def __init__(self, path="", tokenizer=None, max_tokens=1024):
        if not path:
            return
        data = json.load(open(path))
        self.input_ids = []
        self.token_type_ids = []
        self.target = []
        self.ids = []
        self.init_data(data, tokenizer, max_tokens)

    def init_data(self, data, tokenizer, max_tokens):
        #get initial features
        for s_id, d in enumerate(tqdm(data, desc='preprocessing')):
            sentence = d['sentence']
            if 'roberta' in tokenizer.name_or_path:
                for i in range(1, len(sentence)):
                    sentence[i] = ' '+sentence[i]
            sentence1 = [tokenizer.tokenize(w) for w in sentence]
            predicates = d['predicates']
            lemmas = d['lemmas']
            frameset_ids = d['frameset_ids']
            for i in range(len(predicates)):
                pre = predicates[i]
                lemma = lemmas[i]
                gold_fid = frameset_ids[i]
                # some senses in conll2005 are not annotated
                if gold_fid == 'XX':
                    continue
                sentence1i = sentence1[:pre]+[['<p>']] + \
                    sentence1[pre:pre+1]+[['</p>']]+sentence1[pre+1:]
                sentence2i = sum(sentence1i, [])
                if lemma not in lemma_dict:
                    #print('OOV lemma:', lemma)
                    continue
                for fid in lemma_dict[lemma]:
                    q = lemma_dict[lemma][fid]
                    q = tokenizer.tokenize(q)
                    if 'roberta' in tokenizer.name_or_path:
                        txt = ['<s>']+q+['</s>']+['</s>']+sentence2i+['</s>']
                    else:
                        txt = ['[CLS]']+q+['[SEP]']+sentence2i+['[SEP]']
                    txt_ids = tokenizer.convert_tokens_to_ids(txt)
                    txt_ids = torch.tensor(txt_ids, dtype=torch.long)
                    self.input_ids.append(txt_ids)
                    token_type_ids = torch.zeros(
                        txt_ids.shape, dtype=torch.uint8)
                    if 'roberta' in tokenizer.name_or_path:
                        token_type_ids[len(q)+3:] = 1
                    else:
                        token_type_ids[len(q)+2:] = 1
                    self.token_type_ids.append(token_type_ids)
                    target = [1 if fid == gold_fid else 0]
                    self.target.append(target)
                    self.ids.append((s_id, i, lemma, fid))
        self.do_batch(max_tokens)

    def do_batch(self, max_tokens):
        self.batch_input_ids = []
        self.batch_token_type_ids = []
        self.batch_target = []
        self.batch_attention_mask = []
        t = zip(self.input_ids, self.token_type_ids, self.target, self.ids)
        #sort by length
        self.input_ids, self.token_type_ids, self.target, self.ids = zip(
            *sorted(t, key=lambda x:len(x[0])))
        length = [len(i) for i in self.input_ids]
        length = np.array(length)
        #process input that exceeds max tokens 
        length[length > max_tokens] = max_tokens
        indexes = batch_by_tokens(length, max_tokens)
        #batch by tokens        
        for s, e in tqdm(indexes, desc='batching'):
            input_ids = self.input_ids[s:e+1]
            token_type_ids = self.token_type_ids[s:e+1]
            target = self.target[s:e+1]
            input_ids1 = pad_sequence(input_ids, batch_first=True)
            token_type_ids1 = pad_sequence(
                token_type_ids, batch_first=True, padding_value=0)
            target1 = torch.tensor(target, dtype=torch.uint8)
            attention_mask = torch.zeros(input_ids1.shape, dtype=torch.uint8)
            for i in range(input_ids1.shape[0]):
                attention_mask[i, :len(input_ids[i])] = 1
            self.batch_input_ids.append(input_ids1)
            self.batch_token_type_ids.append(token_type_ids1)
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
        token_type_ids = [d.numpy() for d in self.token_type_ids]
        target = np.array(self.target)
        ids = self.ids
        np.save(os.path.join(save_dir, 'input_ids.npy'), input_ids)
        np.save(os.path.join(save_dir, 'token_type_ids.npy'), token_type_ids)
        np.save(os.path.join(save_dir, 'target.npy'), target)
        np.save(os.path.join(save_dir, 'ids.npy'), ids)

    def load(self, save_dir, max_tokens):
        #load cached data
        input_ids = np.load(os.path.join(
            save_dir, 'input_ids.npy'), allow_pickle=True)
        token_type_ids = np.load(os.path.join(
            save_dir, 'token_type_ids.npy'), allow_pickle=True)
        target = np.load(os.path.join(
            save_dir, 'target.npy'), allow_pickle=True)
        ids = np.load(os.path.join(save_dir, 'ids.npy'), allow_pickle=True)
        self.ids = ids.tolist()
        self.input_ids = [torch.from_numpy(d) for d in input_ids]
        self.token_type_ids = [torch.from_numpy(d) for d in token_type_ids]
        self.target = torch.tensor(target)
        self.do_batch(max_tokens)

    def __len__(self):
        return len(self.batch_input_ids)

    def __getitem__(self, i):
        return {'input_ids': self.batch_input_ids[i].long(), 'token_type_ids': self.batch_token_type_ids[i].long(),
                'attention_mask': self.batch_attention_mask[i].float(), 'target': self.batch_target[i].float()}


def load_data(path, pretrained_model_name_or_path, max_tokens, shuffle, local_rank=-1):
    global frames,lemma_dict
    frames_path = os.path.join(os.path.split(path)[0], 'frames.json')
    frames = json.load(open(frames_path))
    lemma_dict = {}
    for k, v in frames.items():
        lemma, fid = k.split('.')
        if lemma not in lemma_dict:
            lemma_dict[lemma] = {}
        lemma_dict[lemma][fid] = v['name']
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
