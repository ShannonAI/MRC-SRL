import os
import copy
import torch
import json
import argparse
import pickle

import spacy
import numpy as np
from tqdm import trange
import Levenshtein
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast

from dataloader import *
from model import MyModel

nlp = spacy.load('en_core_web_sm')

def dis_lemma(lemma):
    '''process lemmas that are not in frame files using edit distance'''
    if lemma not in all_lemmas:
        distances = [Levenshtein.distance(lemma, l) for l in all_lemmas]
        i = np.argmin(distances)
        return all_lemmas[i]
    else:
        return lemma


def lemmatize(sent, predicates, dis=False):
    '''
    lemmatization using spaCy
    Args:
        sent: word list
        predicates: predicate index list
        dis: whether to use edit distance for alignment
    '''
    sent = [s.lower() for s in sent]
    sent1 = spacy.tokens.doc.Doc(nlp.vocab, sent)
    for name, proc in nlp.pipeline:
        sent1 = proc(sent1)
    plemmas = [sent1[p].lemma_ for p in predicates]
    if dis:
        plemmas = [dis_lemma(l) for l in plemmas]
    return plemmas


def load_checkpoint(model_path):
    config = pickle.load(
        open(os.path.join(os.path.split(model_path)[0], 'args'), 'rb'))
    checkpoint = torch.load(model_path)
    model = MyModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return config, model


class DisamDataset:
    def __init__(self, path="", tokenizer=None, max_tokens=1024, lemma_level=0):
        '''
        lemma_level=0: gold lemma
        lemma_level=1: predict lemma without edit distance (used for datasets with many OOV recognized lemmas)
        lemma_level=2: predict lemma with edit distance
        '''
        if not path:
            return
        data = json.load(open(path))
        self.data = copy.deepcopy(data)
        self.input_ids = []
        self.token_type_ids = []
        self.target = []
        self.ids = []  # (sentence id,prediate id,lemma,frameset_id)
        self.lemma_level = lemma_level
        self.init_data(data, tokenizer, max_tokens, lemma_level)

    def init_data(self, data, tokenizer, max_tokens, lemma_level):
        for s_id, d in enumerate(tqdm(data, desc='stage 1')):
            sentence = d['sentence']
            predicates = d['predicates']
            if lemma_level == 0:
                lemmas = d['lemmas']
            elif lemma_level == 1:
                lemmas = lemmatize(sentence, predicates, dis=False)
            elif lemma_level == 2:
                lemmas = lemmatize(sentence, predicates, dis=True)
            else:
                raise Exception()
            if 'roberta' in tokenizer.name_or_path:
                for i in range(1, len(sentence)):
                    sentence[i] = ' '+sentence[i]
            sentence1 = [tokenizer.tokenize(w) for w in sentence]
            frameset_ids = d['frameset_ids']
            for i in range(len(predicates)):
                pre = predicates[i]
                lemma = lemmas[i]
                gold_fid = frameset_ids[i]
                sentence1i = sentence1[:pre]+[['[unused1]']] + \
                    sentence1[pre:pre+1]+[['[unused2]']]+sentence1[pre+1:]
                sentence2i = sum(sentence1i, [])
                if lemma not in lemma_dict:
                    #print('OOV lemma:', lemma)
                    continue
                for fid in lemma_dict[lemma]:
                    q = lemma_dict[lemma][fid]
                    q = tokenizer.tokenize(q)
                    if 'roberta' in tokenizer.name_or_path:
                        txt = ['<s>']+q+['/s']+['<s>']+sentence2i+['/s']
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
        length = [len(i) for i in self.input_ids]
        length = np.array(length)
        length[length > max_tokens] = max_tokens
        indexes = batch_by_tokens(length, max_tokens)
        for s, e in tqdm(indexes, desc='stage2'):
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

    def __len__(self):
        return len(self.batch_input_ids)

    def predict2json(self, predicts, ids):
        # 注意，这里我们的ids
        data = self.data
        if self.lemma_level == 0:
            plemma_ids = 'plemma_ids'  # gold lemma
        else:
            plemma_ids = 'plemma_ids1'  # predict lemma
        for d in data:
            # 01 is the default sense number
            d[plemma_ids] = ['X.01' for _ in d['predicates']]
        for p, _id in zip(predicts, ids):
            s_id, p_id = _id
            data[s_id][plemma_ids][p_id] = p
        return data

    def __getitem__(self, i):
        return {'input_ids': self.batch_input_ids[i].long(), 'token_type_ids': self.batch_token_type_ids[i].long(),
                'attention_mask': self.batch_attention_mask[i].float(), 'target': self.batch_target[i].float()}


def disam_predict(dataset, model, device, amp):
    model.eval()
    targets = []
    predict_probs = []
    ids = dataset.ids
    with torch.no_grad():
        for i in trange(len(dataset), desc='eval'):
            batch = dataset[i]
            input_ids, token_type_ids, attention_mask, target = batch['input_ids'], batch[
                'token_type_ids'], batch['attention_mask'], batch['target']
            input_ids, token_type_ids, attention_mask = input_ids.to(
                device), token_type_ids.to(device), attention_mask.to(device)
            if amp:
                with autocast():
                    predict_prob = model(
                        input_ids, token_type_ids, attention_mask)
            else:
                predict_prob = model(input_ids, token_type_ids, attention_mask)
            targets.append(target.view(-1))
            predict_probs.append(predict_prob)
    targets1 = torch.cat(targets)
    predict_probs1 = torch.cat(predict_probs)
    sents = list(set([(i[0], i[1]) for i in ids]))
    sents_dict = {k: v for v, k in enumerate(sents)}
    label = list(frames.keys())
    lemma_id_dict = {k: v for v, k in enumerate(label)}
    targets2 = torch.zeros([len(sents), len(frames)], dtype=torch.uint8)
    predict_probs2 = torch.zeros([len(sents), len(frames)], dtype=torch.float)
    for t, p, _id in zip(targets1, predict_probs1, ids):
        s_id, p_id, lemma, fid = _id
        id0 = sents_dict[(s_id, p_id)]
        id1 = lemma_id_dict[f'{lemma}.{fid}']
        targets2[id0, id1] = t
        predict_probs2[id0, id1] = p
    predicts = torch.zeros([len(sents), len(frames)],
                           dtype=torch.uint8)  # argmax预测
    argmax_idx = torch.argmax(predict_probs2, dim=-1)
    for i in range(len(predicts)):
        predicts[i][argmax_idx[i]] = 1
    p = (targets2.max(dim=-1)[1] == predicts.max(dim=-1)
         [1]).sum().item()/len(targets2)
    print("score: %.4f" % p)
    predicts1 = []
    for idx in argmax_idx:
        p = label[idx]
        predicts1.append(p)
    return predicts1, sents


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_path')
    parser.add_argument('--dataset_path')
    parser.add_argument('--output_path')
    parser.add_argument('--checkpoint_path')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--lemma_level', type=int, choices=[1, 2])
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--max_tokens', type=int, default=1024)
    args = parser.parse_args()

    frames = json.load(open(args.frames_path))
    all_lemmas = sorted(list(set([k.split('.')[0] for k in frames])))
    lemma_dict = {}
    for k, v in frames.items():
        lemma, fid = k.split('.')
        if lemma not in lemma_dict:
            lemma_dict[lemma] = {}
        lemma_dict[lemma][fid] = v['name']
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    config, model = load_checkpoint(args.checkpoint_path)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        config.pretrained_model_name_or_path)
    #gold lemma
    dataset0 = DisamDataset(args.dataset_path, tokenizer,args.max_tokens, 0)
    print('gold lemma ',end="")
    predicts0, ids0 = disam_predict(dataset0, model, device, args.amp)
    #predict lemma
    dataset1 = DisamDataset(args.dataset_path, tokenizer,args.max_tokens, args.lemma_level)
    print('predict lemma ',end="")
    predicts1, ids1 = disam_predict(dataset1, model, device, args.amp)
    if args.save:
        data0 = dataset0.predict2json(predicts0, ids0)
        data1 = dataset1.predict2json(predicts1, ids1)
        for d0,d1 in zip(data0,data1):
            assert d0['sentence']==d1['sentence'] and d0['predicates']==d1['predicates']
            d0['plemma_ids1']=d1['plemma_ids1']
        with open(args.output_path, 'w') as f:
            json.dump(data0, f, sort_keys=True, indent=4)
