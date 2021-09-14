import time
import os
import random
import argparse

import torch
import pickle
import numpy as np
from transformers.optimization import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import trange, tqdm
from torch.nn.utils import clip_grad_norm_

from model import MyModel
from evaluate import *
from dataloader import *


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset_tag", choices=['conll2005', 'conll2009', 'conll2012'])
    #train_path and dev_path can also be cached data directories.
    parser.add_argument("--train_path")
    parser.add_argument("--dev_path")
    parser.add_argument("--pretrained_model_name_or_path")

    #The specific meanings of arg_query_type, argm_query_type and gold_level are provided in the dataloader
    parser.add_argument("--arg_query_type", type=int, default=2,choices=[0,1,2])
    parser.add_argument("--argm_query_type", type=int, default=1,choices=[0,1])
    parser.add_argument("--gold_level", type=int, choices=[0, 1], default=1)

    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_epochs", default=5, type=int)
    parser.add_argument("--warmup_ratio", type=float, default=-1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1)

    parser.add_argument("--resume", action="store_true",help="used to continue training from the checkpoint")
    parser.add_argument("--checkpoint_path", type=str, help="checkpoint path when resume is true")

    parser.add_argument("--amp", action="store_true", help="whether to enable mixed precision")
    parser.add_argument("--local_rank", type=int, default=-1) #DDP has been implemented but has not been tested.
    parser.add_argument("--eval", action="store_true",help="Whether to evaluate during training")
    parser.add_argument("--tensorboard", action='store_true',help="whether to use tensorboard to log training information")
    parser.add_argument("--save", action="store_true",help="whether to save the trained model")
    parser.add_argument("--tqdm_mininterval", default=1,type=float, help="tqdm minimum update interval")
    args = parser.parse_args()
    return args


def train(args, train_dataloader, dev_dataloader, resume=False, checkpoint=None):
    model = MyModel(args)
    model.train()
    #prepare training
    if args.amp:
        scaler = GradScaler()
        if resume:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
    device = args.local_rank if args.local_rank != -1 else (torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
    if resume:
        model_state_dict = checkpoint['model_state_dict']
        model.load_state_dict(model_state_dict)
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[
                                                          args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], "weight_decay":args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay":0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    if resume:
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        optimizer.load_state_dict(optimizer_state_dict)
    if args.warmup_ratio > 0:
        num_training_steps = len(train_dataloader)*args.max_epochs
        warmup_steps = args.warmup_ratio*num_training_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, num_training_steps)
        if resume:
            scheduler_state_dict = checkpoint['scheduler_state_dict']
            scheduler.load_state_dict(scheduler_state_dict)
    if args.local_rank < 1:
        mid = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
        if resume:
            mid = os.path.split(args.checkpoint_path)[-1]
        if args.tensorboard:
            log_dir = "./logs/{}/arg_labeling/{}".format(args.dataset_tag, mid)
            writer = SummaryWriter(log_dir)
    ltime = time.time()
    start_epoch = 0
    if resume:
        start_epoch = checkpoint['epoch']+1
    #start training  
    for epoch in range(start_epoch, args.max_epochs):
        if args.local_rank != -1:
            train_dataloader.sampler.set_epoch(epoch)
        tqdm_train_dataloader = tqdm(train_dataloader, desc="epoch:%d" % epoch, ncols=150, total=len(
            train_dataloader), mininterval=args.tqdm_mininterval)
        for i, batch in enumerate(tqdm_train_dataloader):
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            input_ids, token_type_ids, attention_mask, target = batch['input_ids'], batch[
                'token_type_ids'], batch['attention_mask'], batch['target']
            input_ids, token_type_ids, attention_mask, target = input_ids.to(
                device), token_type_ids.to(device), attention_mask.to(device), target.to(device)
            if args.amp:
                with autocast():
                    loss = model(input_ids, token_type_ids,
                                 attention_mask, target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.norm(torch.stack(
                    [torch.norm(p.grad) for p in model.parameters() if p.grad is not None]))
                clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model(input_ids, token_type_ids, attention_mask, target)
                loss.backward()
                grad_norm = torch.norm(torch.stack(
                    [torch.norm(p.grad) for p in model.parameters() if p.grad is not None]))
                clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            lr = optimizer.param_groups[0]['lr']
            if args.warmup_ratio > 0:
                scheduler.step()
            if args.local_rank < 1 and args.tensorboard:
                writer.add_scalar('loss', loss.item(), i +
                                  epoch*len(train_dataloader))
                writer.add_scalars(
                    "lr_grad", {"lr": lr, "grad_norm": grad_norm}, i+epoch*len(train_dataloader))
                writer.flush()
            if time.time()-ltime >= args.tqdm_mininterval:
                postfix_str = 'norm:{:.2f},lr:{:.1e},loss:{:.2e}'.format(
                    grad_norm, lr, loss.item())
                tqdm_train_dataloader.set_postfix_str(postfix_str)
                ltime = time.time()
        if args.local_rank < 1 and args.save:
            if hasattr(model, 'module'):
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
            optimizer_state_dict = optimizer.state_dict()
            checkpoint = {}
            checkpoint['model_state_dict'] = model_state_dict
            checkpoint['optimizer_state_dict'] = optimizer_state_dict
            checkpoint = {"model_state_dict": model_state_dict,
                          "optimizer_state_dict": optimizer_state_dict}
            if args.warmup_ratio > 0:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            if args.amp:
                checkpoint["scaler_state_dict"] = scaler.state_dict()
            save_dir = './checkpoints/%s/arg_labeling/%s/' % (args.dataset_tag, mid)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                pickle.dump(args, open(save_dir+'args', 'wb'))
            save_path = save_dir+"checkpoint_%d.cpt" % epoch
            torch.save(checkpoint, save_path)
            print("model saved at:", save_path)
        if args.eval and args.local_rank < 1:
            score = evaluation(model, dev_dataloader, args.amp, device, args.dataset_tag)
            if args.tensorboard:
                hp = vars(args)
                hp['epoch']=epoch
                hp['mid']=mid
                writer.add_hparams(hp,score)
                writer.flush()
            model.train()     
    if args.local_rank < 1 and args.tensorboard:
        writer.close()


if __name__ == "__main__":
    args = args_parser()
    set_seed(args.seed)
    if args.local_rank != -1:
        torch.distributed.init_process_group(backend='nccl')
    if args.train_path.endswith('.json'):
        train_dataloader = load_data(args.train_path, args.pretrained_model_name_or_path, args.max_tokens,
                                     True, args.dataset_tag, args.local_rank, args.gold_level, args.arg_query_type, args.argm_query_type)
        save_dir = args.train_path.replace(
            ".json", '')+f'/arg_labeling/{args.gold_level}_query_type{args.arg_query_type}.{args.argm_query_type}/'+args.pretrained_model_name_or_path.split('/')[-1]
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        train_dataloader.dataset.save(save_dir)
        print('training data saved at:', save_dir)
    else:
        train_dataloader = reload_data(
            args.train_path, args.max_tokens, True, args.local_rank)
    if not args.eval:
        dev_dataloader = None
    elif args.dev_path.endswith('.json'):
        dev_dataloader = load_data(args.dev_path, args.pretrained_model_name_or_path, args.max_tokens, False,
                                   args.dataset_tag, args.local_rank, args.gold_level, args.arg_query_type, args.argm_query_type)
        save_dir = args.dev_path.replace(
            ".json", '')+f'/arg_labeling/{args.gold_level}_query_type{args.arg_query_type}.{args.argm_query_type}/'+args.pretrained_model_name_or_path.split('/')[-1]
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        dev_dataloader.dataset.save(save_dir)
        print('validation data saved at:', save_dir)
    else:
        dev_dataloader = reload_data(args.dev_path, args.max_tokens, False, -1)
    print(args)
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.checkpoint_path,
                                map_location=torch.device('cpu'))
        checkpoint['epoch'] = int(
            args.checkpoint_path.split('_')[-1].split('.')[0])
    train(args, train_dataloader, dev_dataloader, args.resume, checkpoint)
