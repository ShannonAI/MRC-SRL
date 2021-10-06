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


def lemmatize(sent, predicates, dis=True):
    '''
    lemmatization using spaCy, you can replace it with other lemmatizers.
    Args:
        sent: word list
        predicates: predicate index list
        dis: use edit distance
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
    def __init__(self, path="", tokenizer=None, max_tokens=1024):
        if not path:
            return
        data = json.load(open(path))
        self.data = copy.deepcopy(data)
        self.input_ids = []
        self.token_type_ids = []
        self.target = []
        self.ids = []  # (sentence id,prediate id,lemma,frameset_id)
        self.init_data(data, tokenizer, max_tokens)

    def init_data(self, data, tokenizer, max_tokens):
        for s_id, d in enumerate(tqdm(data, desc='preprocessing')):
            sentence = d['sentence']
            predicates = d['predicates']
            glemmas = d['lemmas']
            plemmas = lemmatize(sentence, predicates)
            plemmas1 = lemmatize(sentence, predicates,False) #don't use edit distance
            if 'roberta' in tokenizer.name_or_path:
                for i in range(1, len(sentence)):
                    sentence[i] = ' '+sentence[i]
            sentence1 = [tokenizer.tokenize(w) for w in sentence]
            frameset_ids = d['frameset_ids']
            for i in range(len(predicates)):
                pre = predicates[i]
                glemma = glemmas[i]
                plemma = plemmas[i]
                plemma1 = plemmas1[i]
                gold_fid = frameset_ids[i]
                sentence1i = sentence1[:pre]+[['<p>']] + \
                    sentence1[pre:pre+1]+[['</p>']]+sentence1[pre+1:]
                sentence2i = sum(sentence1i, [])
                for fid in lemma_dict[plemma]:
                    q = lemma_dict[plemma][fid]
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
                    self.ids.append((s_id, i, glemma, plemma, fid, plemma1)) #meta information about this sample
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

    def __len__(self):
        return len(self.batch_input_ids)

    def predict2json(self, predicts, ids):
        for d in self.data:
            # 01 is the default sense number
            d['plemma_ids'] = ['X.01' for _ in d['predicates']]        
        for p, _id in zip(predicts, ids):
            s_id, p_id = _id
            self.data[s_id]['plemma_ids'][p_id] = p
        return self.data

    def __getitem__(self, i):
        return {'input_ids': self.batch_input_ids[i].long(), 'token_type_ids': self.batch_token_type_ids[i].long(),
                'attention_mask': self.batch_attention_mask[i].float(), 'target': self.batch_target[i].float()}

def disam_predict(dataset, model, device, amp):
    '''
    Note that due to the existence of OOV gold lemma, the evaluation of predicate disambiguation does not use edit distance,
    but the predicate disambiguation results used for argument labeling use the edit distance to address the OOV problem.
    '''
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
    sp_ids = list(set([(i[0], i[1]) for i in ids]))
    sp_dict = {k: v for v, k in enumerate(sp_ids)}
    metas = {k:[] for k in sp_ids}
    lemma_eval = [[] for _ in sp_ids]
    fid_eval = [[] for _ in sp_ids]
    for t,p,_id in zip(targets1,predict_probs1,ids):
        s_id, p_id, glemma, plemma, fid, plemma1 = _id
        lemma_eval[sp_dict[(s_id,p_id)]].append([1 if glemma==plemma else 0,1 if glemma==plemma1 else 0])
        metas[(s_id,p_id)].append((glemma, plemma, fid, plemma1))        
        fid_eval[sp_dict[(s_id,p_id)]].append((t,p))
    lemma_eval1 = []
    for le in lemma_eval:
        w_dis = max([i[0] for i in le])
        wo_dis = max([i[1] for i in le])
        lemma_eval1.append((w_dis,wo_dis)) 
    fid_eval1 = []
    for i,fe in enumerate(fid_eval):
        gold_idx = np.argmax([f[0] for f in fe])
        pre_idx = np.argmax([f[1] for f in fe])
        # 01 is the default sense number for OOV lemma
        pre_wo_dis = '01'==metas[sp_ids[i]][gold_idx][2] if metas[sp_ids[i]][0][-1] not in lemma_dict else gold_idx==pre_idx
        fid_eval1.append((gold_idx==pre_idx,pre_wo_dis))        
    #print("lemma recognition accuracy: %.4f"%((sum([i[0] for i in lemma_eval1]))/len(lemma_eval1)))
    #print("lemma recognition accuracy1: %.4f"%((sum([i[1] for i in lemma_eval1]))/len(lemma_eval1)))
    #print("sense recognition accuracy: %.4f"%((sum([i[0] for i in fid_eval1]))/len(fid_eval1)))
    #print("sense recognition accuracy1: %.4f"%((sum([i[1] for i in fid_eval1]))/len(fid_eval1)))    
    #print("with edit distance accuracy: %.4f"%((sum([i[0]*j[0] for i,j in zip(lemma_eval1,fid_eval1)]))/len(fid_eval1)))
    #print("without edit distance accuracy: %.4f"%((sum([i[1]*j[1] for i,j in zip(lemma_eval1,fid_eval1)]))/len(fid_eval1)))
    print("accuracy: %.4f"%((sum([i[1]*j[1] for i,j in zip(lemma_eval1,fid_eval1)]))/len(fid_eval1)))    
    predicts1 = []
    for i,fe in enumerate(fid_eval):
        k=sp_ids[i]
        meta = metas[k]
        pre_idx = np.argmax([i[1] for i in fe])
        glemma, plemma, fid, plemma1 = meta[pre_idx]
        predicts1.append(f'{plemma}.{fid}')
    return predicts1, sp_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_path')
    parser.add_argument('--dataset_path')
    parser.add_argument('--output_path')
    parser.add_argument('--checkpoint_path')
    parser.add_argument('--save', action='store_true')
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
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name_or_path)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<p>', '</p>']})
    dataset = DisamDataset(args.dataset_path, tokenizer,args.max_tokens)
    predicts, ids = disam_predict(dataset, model, device, args.amp)
    if args.save:
        data = dataset.predict2json(predicts, ids)
        with open(args.output_path, 'w') as f:
            json.dump(data, f, sort_keys=True, indent=4)
