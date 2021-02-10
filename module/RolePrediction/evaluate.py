import torch
from tqdm import tqdm
from torch.cuda.amp import autocast


def evaluation(model,dataloader,amp=False,device=torch.device('cpu')):
    if hasattr(model,'module'):  
        model = model.module
    model.eval()
    model.to(device)
    tqdm_dataloader = tqdm(dataloader,desc="dev eval")
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
    gold1 = torch.cat(gold,dim=0).view(-1)
    predict_probs1 = torch.cat(predict_probs,dim=0)
    predict = (predict_probs1>0.5).int().view(-1)
    score = sum([i==j for i,j in zip(predict,gold1)])/len(predict)
    print('score: %.4f'%score)