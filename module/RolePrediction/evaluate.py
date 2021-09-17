import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
from sklearn.metrics import classification_report


def evaluation(model,dataloader,amp=False,device=torch.device('cpu'),dataset_tag=""):
    if dataset_tag == 'conll2005' or dataset_tag == 'conll2009':
        ARGS = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'AA']
        ARGMS = ['DIR', 'LOC', 'MNR', 'TMP', 'EXT', 'REC',
                 'PRD', 'PNC', 'CAU', 'DIS', 'ADV', 'MOD', 'NEG']
    elif dataset_tag == 'conll2012':
        ARGS = ['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5', 'ARGA']
        ARGMS = ['MNR', 'ADV', 'LOC', 'TMP', 'PRP', 'PRD', 'DIR', 'DIS', 'MOD',
                 'NEG', 'CAU', 'EXT', 'LVB', 'REC', 'ADJ', 'GOL', 'DSP', 'PRR', 'COM', 'PRX', 'PNC']
    else:
        raise Exception("Invalid Dataset Tag:%s" % dataset_tag)    
    if hasattr(model,'module'):  
        model = model.module
    model.eval()
    model.to(device)
    tqdm_dataloader = tqdm(dataloader,desc="eval")
    gold = []
    predict_probs = []
    with torch.no_grad():
        for i,batch in enumerate(tqdm_dataloader):
            input_ids,attention_mask,target=batch['input_ids'],batch['attention_mask'],batch['target']
            input_ids,attention_mask=input_ids.to(device),attention_mask.to(device)
            if amp:
                with autocast():
                    predict_prob = model(input_ids,attention_mask)
            else:
                predict_prob = model(input_ids,attention_mask)
            gold.append(target)
            predict_probs.append(predict_prob)
    gold1 = torch.cat(gold,dim=0)
    predict_probs1 = torch.cat(predict_probs,dim=0)
    predict = (predict_probs1>0.5).int()
    score = classification_report(gold1,predict,output_dict=True,zero_division=0,target_names=ARGS+ARGMS)['micro avg']
    p,r,f = score['precision'],score['recall'],score['f1-score']
    print('micro avg score: p:{:.4f} r:{:.4f} f:{:.4f}'.format(p,r,f))
    return {'p':p,'r':r,'f':f}
