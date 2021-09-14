import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
from dataloader import *


def get_score(gold_set, predict_set):
    #print("len gold",len(gold_set),"len predict",len(predict_set))
    TP = len(set.intersection(gold_set, predict_set))
    precision = TP/(len(predict_set)+1e-6)
    recall = TP/(len(gold_set)+1e-6)
    f1 = 2*precision*recall/(precision+recall+1e-6)
    return precision, recall, f1


def evaluation(model, dataloader, amp=False, device=torch.device('cpu')):
    # Note this evaluation may be not completely correct, since we skip some samples without sense annonatations or with OOV lemmas
    if hasattr(model, 'module'):
        model = model.module
    model.eval()
    model.to(device)
    tqdm_dataloader = tqdm(dataloader, desc="eval")
    targets = []
    predict_probs = []
    ids = dataloader.dataset.ids
    with torch.no_grad():
        for i, batch in enumerate(tqdm_dataloader):
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
    label = list(sorted([f'{i[2]}.{i[3]}' for i in ids]))
    lemma_id_dict = {k: v for v, k in enumerate(label)}
    targets2 = torch.zeros([len(sents), len(label)], dtype=torch.uint8)
    predict_probs2 = torch.zeros([len(sents), len(label)], dtype=torch.float)
    for t, p, _id in zip(targets1, predict_probs1, ids):
        s_id, p_id, lemma, fid = _id
        id0 = sents_dict[(s_id, p_id)]
        id1 = lemma_id_dict[f'{lemma}.{fid}']
        targets2[id0, id1] = t
        predict_probs2[id0, id1] = p
    predicts = torch.zeros([len(sents), len(label)], dtype=torch.uint8)
    argmax_idx = torch.argmax(predict_probs2, dim=-1)
    for i in range(len(predicts)):
        predicts[i][argmax_idx[i]] = 1
    p = (targets2.max(dim=-1)[1] == predicts.max(dim=-1)[1]).sum().item()/len(targets2)
    print("acccuracy score: %.4f" % p)
    return {"accuracy":p}
