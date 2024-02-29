import numpy as np
import random
import os
import torch
# import pickle
import time
from collections import defaultdict
from dataset_seq import *
#from dataset import *
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import argparse
import torch.nn.functional as F
#from model import *
from model_seq import *
from sklearn.metrics import roc_auc_score
from pathlib import Path
# from pypai.model import upload_model
from tqdm import tqdm
from functools import partial
import logging
from utils import *
# from thop import profile
from sklearn.model_selection import train_test_split  # 划分数据集

logger = logging.getLogger()

def test(model,args,valLoader):
    model.eval()
    stats = AverageMeter('loss','loss_cls')
    # stats = AverageMeter('loss','ndcg_1_d1','ndcg_5_d1','ndcg_10_d1','ndcg_1_d2','ndcg_5_d2','ndcg_10_d2','hit_1_d1','hit_5_d1','hit_10_d1','hit_1_d2','hit_5_d2','hit_10_d2','MRR_d1','MRR_d2')
    pred_d1_list = None
    pred_d2_list = None
    pred_d1_list_cs = None
    pred_d2_list_cs = None
    pred_d1_list_tailed = None
    pred_d2_list_tailed = None
    criterion_cls = nn.BCELoss(reduce=False)
    fix_value = 1e-7 # fix the same value 
    for k,sample in enumerate(tqdm(valLoader)):
        u_node = torch.LongTensor(sample['user_node'].long()).cuda()
        i_node = torch.LongTensor(sample['i_node'].long()).cuda()
        neg_samples = torch.LongTensor(sample['neg_samples'].long()).cuda()
        seq_d1 = torch.LongTensor(sample['seq_d1'].long()).cuda()
        seq_d2 = torch.LongTensor(sample['seq_d2'].long()).cuda()
        also_d1_view = torch.LongTensor(sample['also_d1_view'].long()).cuda()
        also_d2_view = torch.LongTensor(sample['also_d2_view'].long()).cuda()
        domain_id = torch.LongTensor(sample['domain_id'].long()).cuda()
        overlap_label = torch.LongTensor(sample['overlap_label'].long()).cuda()
        tailed_label = torch.LongTensor(sample['tailed_label'].long()).cuda()
        cs_label = torch.LongTensor(sample['cs_label'].long()).cuda()
        labels = torch.LongTensor(sample['label'].long()).cuda()
        labels = labels.float()
        with torch.no_grad():
            if args.isCL:
                predict_d1, predict_d2, s14_false, s23_false, s12_true, s34_true = model(u_node,i_node,neg_samples,seq_d1,seq_d2,also_d1_view,also_d2_view,cs_label,False)
            else:
                predict_d1, predict_d2 = model(u_node,i_node,neg_samples,seq_d1,seq_d2,also_d1_view,also_d2_view,cs_label,False)
            # predict_d1, predict_d2,u_feat_enhance_m1_d1, u_feat_enhance_m1_d2, u_feat_enhance_m2_d1,u_feat_enhance_m2_d2, u_feat_enhance_m3_d1,u_feat_enhance_m3_d2, u_feat_enhance_m4_d1, u_feat_enhance_m4_d2 = model(u_node,i_node,neg_samples,seq_d1,seq_d2,long_tail_mask_d1,long_tail_mask_d2)
        predict_d1 = predict_d1.squeeze()
        predict_d2 = predict_d2.squeeze()
        one_value = torch.LongTensor(torch.ones(domain_id.shape[0]).long()).cuda()
        mask_d1 = torch.LongTensor((one_value.cpu() - domain_id.cpu()).long()).cuda()
        mask_d2 = torch.LongTensor((domain_id.cpu()).long()).cuda()
        loss_cls = criterion_cls(predict_d1,labels) * mask_d1.unsqueeze(1) + criterion_cls(predict_d2,labels) * mask_d2.unsqueeze(1)
        loss_cls = torch.mean(loss_cls)
        # label_domain = torch.LongTensor([0,1,0,1]).cuda().float()
        # loss_cl = nn.BCELoss()(predict_domain.squeeze(),label_domain)
        # loss_cl =  cal_loss_cl_refine(u_feat_enhance_m1_d1,u_feat_enhance_m4_d1)+cal_loss_cl_refine(u_feat_enhance_m1_d2,u_feat_enhance_m4_d2)
        loss = loss_cls #+ loss_cl * 0.05
        stats.update(loss=loss.item(),loss_cls=loss_cls.item())#,loss_cl=loss_cl.item())
        domain_id = domain_id.unsqueeze(1).expand_as(predict_d1)
        overlap_label = overlap_label.unsqueeze(1).expand_as(predict_d1)
        tailed_label = tailed_label.unsqueeze(1).expand_as(predict_d1)
        cs_label = cs_label.unsqueeze(1).expand_as(predict_d1)
        predict_d1 = predict_d1.view(-1,args.neg_nums+1).cpu().detach().numpy().copy()
        predict_d2 = predict_d2.view(-1,args.neg_nums+1).cpu().detach().numpy().copy()
        domain_id = domain_id.view(-1,args.neg_nums+1).cpu().detach().numpy().copy()
        if args.overlap:
            tailed_label = tailed_label.view(-1,args.neg_nums+1).cpu().detach().numpy().copy()
            cs_label = cs_label.view(-1,args.neg_nums+1).cpu().detach().numpy().copy()
            predict_d1_cse_cs, _3, predict_d2_cse_cs, _4 = choose_predict_overlap(predict_d1,predict_d2,domain_id,cs_label)
            if pred_d1_list_cs is None and not isinstance(predict_d1_cse_cs,list):
                pred_d1_list_cs = predict_d1_cse_cs
            elif pred_d1_list_cs is not None and not isinstance(predict_d1_cse_cs,list):
                pred_d1_list_cs = np.append(pred_d1_list_cs, predict_d1_cse_cs, axis=0)
            if pred_d2_list_cs is None and not isinstance(predict_d2_cse_cs,list):
                pred_d2_list_cs = predict_d2_cse_cs
            elif pred_d2_list_cs is not None and not isinstance(predict_d2_cse_cs,list):
                pred_d2_list_cs = np.append(pred_d2_list_cs, predict_d2_cse_cs, axis=0)
            predict_d1_cse_tail, _5, predict_d2_cse_tail, _6 = choose_predict_overlap(predict_d1,predict_d2,domain_id,tailed_label)
            if pred_d1_list_tailed is None and not isinstance(predict_d1_cse_tail,list):
                pred_d1_list_tailed = predict_d1_cse_tail
            elif pred_d1_list_tailed is not None and not isinstance(predict_d1_cse_tail,list):
                pred_d1_list_tailed = np.append(pred_d1_list_tailed, predict_d1_cse_tail, axis=0)
            if pred_d2_list_tailed is None and not isinstance(predict_d2_cse_tail,list):
                pred_d2_list_tailed = predict_d2_cse_tail
            elif pred_d2_list_tailed is not None and not isinstance(predict_d2_cse_tail,list):
                pred_d2_list_tailed = np.append(pred_d2_list_tailed, predict_d2_cse_tail, axis=0)
        predict_d1_cse, predict_d2_cse = choose_predict(predict_d1,predict_d2,domain_id)
        if pred_d1_list is None and not isinstance(predict_d1_cse,list):
            pred_d1_list = predict_d1_cse
        elif pred_d1_list is not None and not isinstance(predict_d1_cse,list):
            pred_d1_list = np.append(pred_d1_list, predict_d1_cse, axis=0)
        if pred_d2_list is None and not isinstance(predict_d2_cse,list):
            pred_d2_list = predict_d2_cse
        elif pred_d2_list is not None and not isinstance(predict_d2_cse,list):
            pred_d2_list = np.append(pred_d2_list, predict_d2_cse, axis=0)

    if not args.overlap:        
        pred_d1_list[:,0] = pred_d1_list[:,0]-fix_value
        pred_d2_list[:,0] = pred_d2_list[:,0]-fix_value
        HIT_1_d1, NDCG_1_d1, HIT_5_d1, NDCG_5_d1, HIT_10_d1, NDCG_10_d1, MRR_d1 = get_sample_scores(pred_d1_list)
        HIT_1_d2, NDCG_1_d2, HIT_5_d2, NDCG_5_d2, HIT_10_d2, NDCG_10_d2, MRR_d2 = get_sample_scores(pred_d2_list)
        return stats.loss, stats.loss_cls, HIT_1_d1, NDCG_1_d1, HIT_5_d1, NDCG_5_d1, HIT_10_d1, NDCG_10_d1, MRR_d1, HIT_1_d2, NDCG_1_d2, HIT_5_d2, NDCG_5_d2, HIT_10_d2, NDCG_10_d2, MRR_d2
    else:
        HIT_1_d1_cs, NDCG_1_d1_cs, HIT_5_d1_cs, NDCG_5_d1_cs, HIT_10_d1_cs, NDCG_10_d1_cs, MRR_d1_cs = get_sample_scores(pred_d1_list_cs)
        HIT_1_d2_cs, NDCG_1_d2_cs, HIT_5_d2_cs, NDCG_5_d2_cs, HIT_10_d2_cs, NDCG_10_d2_cs, MRR_d2_cs = get_sample_scores(pred_d2_list_cs)

        HIT_1_d1_tailed, NDCG_1_d1_tailed, HIT_5_d1_tailed, NDCG_5_d1_tailed, HIT_10_d1_tailed, NDCG_10_d1_tailed, MRR_d1_tailed = get_sample_scores(pred_d1_list_tailed)
        HIT_1_d2_tailed, NDCG_1_d2_tailed, HIT_5_d2_tailed, NDCG_5_d2_tailed, HIT_10_d2_tailed, NDCG_10_d2_tailed, MRR_d2_tailed = get_sample_scores(pred_d2_list_tailed)
        pred_d1_list[:,0] = pred_d1_list[:,0]-fix_value
        pred_d2_list[:,0] = pred_d2_list[:,0]-fix_value
        HIT_1_d1, NDCG_1_d1, HIT_5_d1, NDCG_5_d1, HIT_10_d1, NDCG_10_d1, MRR_d1 = get_sample_scores(pred_d1_list)
        HIT_1_d2, NDCG_1_d2, HIT_5_d2, NDCG_5_d2, HIT_10_d2, NDCG_10_d2, MRR_d2 = get_sample_scores(pred_d2_list)
        return stats.loss, stats.loss_cls, HIT_1_d1_cs, NDCG_1_d1_cs, HIT_5_d1_cs, NDCG_5_d1_cs, HIT_10_d1_cs, NDCG_10_d1_cs, MRR_d1_cs, HIT_1_d2_cs, NDCG_1_d2_cs, HIT_5_d2_cs, NDCG_5_d2_cs, HIT_10_d2_cs, NDCG_10_d2_cs, MRR_d2_cs, HIT_1_d1_tailed, NDCG_1_d1_tailed, HIT_5_d1_tailed, NDCG_5_d1_tailed, HIT_10_d1_tailed, NDCG_10_d1_tailed, MRR_d1_tailed, HIT_1_d2_tailed, NDCG_1_d2_tailed, HIT_5_d2_tailed, NDCG_5_d2_tailed, HIT_10_d2_tailed, NDCG_10_d2_tailed, MRR_d2_tailed, HIT_1_d1, NDCG_1_d1, HIT_5_d1, NDCG_5_d1, HIT_10_d1, NDCG_10_d1, MRR_d1, HIT_1_d2, NDCG_1_d2, HIT_5_d2, NDCG_5_d2, HIT_10_d2, NDCG_10_d2, MRR_d2

def train(model,trainLoader,args,valLoader):
    best_hit_1_d1 = 0
    best_hit_5_d1 = 0
    best_hit_10_d1 = 0
    best_hit_1_d2 = 0
    best_hit_5_d2 = 0
    best_hit_10_d2 = 0

    best_ndcg_1_d1 = 0
    best_ndcg_5_d1 = 0
    best_ndcg_10_d1 = 0
    best_ndcg_1_d2 = 0
    best_ndcg_5_d2 = 0
    best_ndcg_10_d2 = 0

    best_mrr_d1 = 0
    best_mrr_d2 = 0

    best_hit_1_d1_tailed = 0
    best_hit_5_d1_tailed = 0
    best_hit_10_d1_tailed = 0
    best_hit_1_d2_tailed = 0
    best_hit_5_d2_tailed = 0
    best_hit_10_d2_tailed = 0

    best_ndcg_1_d1_tailed = 0
    best_ndcg_5_d1_tailed = 0
    best_ndcg_10_d1_tailed = 0
    best_ndcg_1_d2_tailed = 0
    best_ndcg_5_d2_tailed = 0
    best_ndcg_10_d2_tailed = 0

    best_mrr_d1_tailed = 0
    best_mrr_d2_tailed = 0

    best_hit_1_d1_cs = 0
    best_hit_5_d1_cs = 0
    best_hit_10_d1_cs = 0
    best_hit_1_d2_cs = 0
    best_hit_5_d2_cs = 0
    best_hit_10_d2_cs = 0

    best_ndcg_1_d1_cs = 0
    best_ndcg_5_d1_cs = 0
    best_ndcg_10_d1_cs = 0
    best_ndcg_1_d2_cs = 0
    best_ndcg_5_d2_cs = 0
    best_ndcg_10_d2_cs = 0

    best_mrr_d1_cs = 0
    best_mrr_d2_cs = 0
    save_path1 = Path(args.model_dir) / 'checkpoint' / 'best_d1.pt'
    save_path2 = Path(args.model_dir) / 'checkpoint' / 'best_d2.pt'
    criterion_recon = partial(sce_loss, alpha=args.alpha_l)
    criterion_cls = nn.BCELoss(reduce=False)
    if not os.path.exists(os.path.join(Path(args.model_dir),'checkpoint')):
        os.mkdir(os.path.join(Path(args.model_dir),'checkpoint'))
    for epoch in range(args.epoch):
        stats = AverageMeter('loss','loss_cls','loss_cls_m1','loss_cls_m2','loss_cls_m3')
        model.train()
        for i,sample in enumerate(tqdm(trainLoader)):
            u_node = torch.LongTensor(sample['user_node'].long()).cuda()
            i_node = torch.LongTensor(sample['i_node'].long()).cuda()
            neg_samples = torch.LongTensor(sample['neg_samples'].long()).cuda()
            seq_d1 = torch.LongTensor(sample['seq_d1'].long()).cuda()
            also_view = torch.LongTensor(sample['also_view'].long()).cuda()
            seq_d2 = torch.LongTensor(sample['seq_d2'].long()).cuda()
            also_d1_view = torch.LongTensor(sample['also_d1_view'].long()).cuda()
            also_d2_view = torch.LongTensor(sample['also_d2_view'].long()).cuda()
            domain_id = torch.LongTensor(sample['domain_id'].long()).cuda()
            labels = torch.LongTensor(sample['label'].long()).cuda()
            labels = labels.float()
            if args.isCL:
                predict_d1, predict_d2, s14_false, s23_false, s12_true, s34_true = model(u_node,i_node,neg_samples,seq_d1,seq_d2,also_d1_view,also_d2_view)
            else:
                predict_d1, predict_d2 = model(u_node,i_node,neg_samples,seq_d1,seq_d2,also_d1_view,also_d2_view)
            predict_d1 = predict_d1.squeeze()
            predict_d2 = predict_d2.squeeze()
            one_value = torch.LongTensor(torch.ones(domain_id.shape[0]).long()).cuda()
            mask_d1 = torch.LongTensor((one_value.cpu() - domain_id.cpu()).long()).cuda()
            mask_d2 = torch.LongTensor((domain_id.cpu()).long()).cuda()
            # print(predict_d1.shape)
            # print(labels.shape)
            loss_cls = criterion_cls(predict_d1,labels) * mask_d1.unsqueeze(1)  + criterion_cls(predict_d2,labels) * mask_d2.unsqueeze(1) #* 2
            loss_cls = torch.mean(loss_cls)
            if args.isCL:
                true_label = torch.LongTensor(torch.ones_like(s12_true.cpu()).long()).cuda().float()
                false_label = torch.LongTensor(torch.zeros_like(s14_false.cpu()).long()).cuda().float()
                loss = loss_cls + (nn.BCELoss()(s12_true,true_label)+nn.BCELoss()(s34_true,true_label)+nn.BCELoss()(s14_false,false_label)+nn.BCELoss()(s23_false,false_label))*args.CL_w
            else:
                loss = loss_cls
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # auc_avg_domain = roc_auc_score(labels.detach().cpu().numpy(),predict.detach().cpu().numpy())
            stats.update(loss=loss.item(),loss_cls=loss_cls.item())
            if i % 20 == 0:
                logger.info(f'train total loss:{stats.loss}, cls loss:{stats.loss_cls} \t')
            #print("epoch :{} train loss:{}, auc:{}".format(epoch,stats.loss,stats.auc)) 
        #val(epoch)
        if args.overlap:
            val_loss, val_cls_loss, HIT_1_d1_cs, NDCG_1_d1_cs, HIT_5_d1_cs, NDCG_5_d1_cs, HIT_10_d1_cs, NDCG_10_d1_cs, MRR_d1_cs, HIT_1_d2_cs, NDCG_1_d2_cs, HIT_5_d2_cs, NDCG_5_d2_cs, HIT_10_d2_cs, NDCG_10_d2_cs, MRR_d2_cs, HIT_1_d1_tailed, NDCG_1_d1_tailed, HIT_5_d1_tailed, NDCG_5_d1_tailed, HIT_10_d1_tailed, NDCG_10_d1_tailed, MRR_d1_tailed, HIT_1_d2_tailed, NDCG_1_d2_tailed, HIT_5_d2_tailed, NDCG_5_d2_tailed, HIT_10_d2_tailed, NDCG_10_d2_tailed, MRR_d2_tailed, HIT_1_d1, NDCG_1_d1, HIT_5_d1, NDCG_5_d1, HIT_10_d1, NDCG_10_d1, MRR_d1, HIT_1_d2, NDCG_1_d2, HIT_5_d2, NDCG_5_d2, HIT_10_d2, NDCG_10_d2, MRR_d2  = test(model,args,valLoader)

            best_hit_1_d1_tailed = max(HIT_1_d1_tailed,best_hit_1_d1_tailed)
            best_hit_5_d1_tailed = max(HIT_5_d1_tailed,best_hit_5_d1_tailed)
            best_hit_10_d1_tailed = max(HIT_10_d1_tailed,best_hit_10_d1_tailed)
            best_hit_1_d2_tailed = max(HIT_1_d2_tailed,best_hit_1_d2_tailed)
            best_hit_5_d2_tailed = max(HIT_5_d2_tailed,best_hit_5_d2_tailed)
            best_hit_10_d2_tailed = max(HIT_10_d2_tailed,best_hit_10_d2_tailed)

            best_ndcg_1_d1_tailed = max(best_ndcg_1_d1_tailed,NDCG_1_d1_tailed)
            best_ndcg_5_d1_tailed = max(best_ndcg_5_d1_tailed,NDCG_5_d1_tailed)
            best_ndcg_10_d1_tailed = max(best_ndcg_10_d1_tailed,NDCG_10_d1_tailed)
            best_ndcg_1_d2_tailed = max(best_ndcg_1_d2_tailed,NDCG_1_d2_tailed)
            best_ndcg_5_d2_tailed = max(best_ndcg_5_d2_tailed,NDCG_5_d2_tailed)
            best_ndcg_10_d2_tailed = max(best_ndcg_10_d2_tailed,NDCG_10_d2_tailed)
            best_mrr_d1_tailed = max(best_mrr_d1_tailed,MRR_d1_tailed)
            best_mrr_d2_tailed = max(best_mrr_d2_tailed,MRR_d2_tailed)   

            best_hit_1_d1_cs = max(HIT_1_d1_cs,best_hit_1_d1_cs)
            best_hit_5_d1_cs = max(HIT_5_d1_cs,best_hit_5_d1_cs)
            best_hit_10_d1_cs = max(HIT_10_d1_cs,best_hit_10_d1_cs)
            best_hit_1_d2_cs = max(HIT_1_d2_cs,best_hit_1_d2_cs)
            best_hit_5_d2_cs = max(HIT_5_d2_cs,best_hit_5_d2_cs)
            best_hit_10_d2_cs = max(HIT_10_d2_cs,best_hit_10_d2_cs)

            best_ndcg_1_d1_cs = max(best_ndcg_1_d1_cs,NDCG_1_d1_cs)
            best_ndcg_5_d1_cs = max(best_ndcg_5_d1_cs,NDCG_5_d1_cs)
            best_ndcg_10_d1_cs = max(best_ndcg_10_d1_cs,NDCG_10_d1_cs)
            best_ndcg_1_d2_cs = max(best_ndcg_1_d2_cs,NDCG_1_d2_cs)
            best_ndcg_5_d2_cs = max(best_ndcg_5_d2_cs,NDCG_5_d2_cs)
            best_ndcg_10_d2_cs = max(best_ndcg_10_d2_cs,NDCG_10_d2_cs)
            best_mrr_d1_cs = max(best_mrr_d1_cs,MRR_d1_cs)
            best_mrr_d2_cs = max(best_mrr_d2_cs,MRR_d2_cs)       

            best_hit_1_d1 = max(HIT_1_d1,best_hit_1_d1)
            best_hit_5_d1 = max(HIT_5_d1,best_hit_5_d1)
            best_hit_10_d1 = max(HIT_10_d1,best_hit_10_d1)
            best_hit_1_d2 = max(HIT_1_d2,best_hit_1_d2)
            best_hit_5_d2 = max(HIT_5_d2,best_hit_5_d2)
            best_hit_10_d2 = max(HIT_10_d2,best_hit_10_d2)

            best_ndcg_1_d1 = max(best_ndcg_1_d1,NDCG_1_d1)
            best_ndcg_5_d1 = max(best_ndcg_5_d1,NDCG_5_d1)
            best_ndcg_10_d1 = max(best_ndcg_10_d1,NDCG_10_d1)
            best_ndcg_1_d2 = max(best_ndcg_1_d2,NDCG_1_d2)
            best_ndcg_5_d2 = max(best_ndcg_5_d2,NDCG_5_d2)
            best_ndcg_10_d2 = max(best_ndcg_10_d2,NDCG_10_d2)
            best_mrr_d1 = max(best_mrr_d1,MRR_d1)
            best_mrr_d2 = max(best_mrr_d2,MRR_d2)
            logger.info(f'Epoch: {epoch}/{args.epoch} \t'
                        f'Train Loss: {stats.loss:.4f} \t'
                        f'Val loss: {val_loss:.4f}, cls loss: {val_cls_loss:.4f}\n'
                        f'val domain1 cur/max tailed users HR@1: {HIT_1_d1_tailed:.4f}/{best_hit_1_d1_tailed:.4f} \n,' 
                        f'tailed users HR@5: {HIT_5_d1_tailed:.4f}/{best_hit_5_d1_tailed:.4f} \n, '
                        f'tailed users HR@10: {HIT_10_d1_tailed:.4f}/{best_hit_10_d1_tailed:.4f} \n'
                        f'tailed users NDCG@5: {NDCG_5_d1_tailed:.4f}/{best_ndcg_5_d1_tailed:.4f} \n, '
                        f'tailed users NDCG@10: {NDCG_10_d1_tailed:.4f}/{best_ndcg_10_d1_tailed:.4f}, \n'
                        f'tailed users MRR: {MRR_d1_tailed:.4f}/{best_mrr_d1_tailed:.4f} \n'
                        f'val domain1 cur/max cold-started users HR@1: {HIT_1_d1_cs:.4f}/{best_hit_1_d1_cs:.4f} \n,' 
                        f'cold-started users HR@5: {HIT_5_d1_cs:.4f}/{best_hit_5_d1_cs:.4f} \n, '
                        f'cold-started users HR@10: {HIT_10_d1_cs:.4f}/{best_hit_10_d1_cs:.4f} \n'
                        f'cold-started users NDCG@5: {NDCG_5_d1_cs:.4f}/{best_ndcg_5_d1_cs:.4f} \n, '
                        f'cold-started users NDCG@10: {NDCG_10_d1_cs:.4f}/{best_ndcg_10_d1_cs:.4f}, \n'
                        f'cold-started users MRR: {MRR_d1_cs:.4f}/{best_mrr_d1_cs:.4f} \n'
                        f'val domain2 cur/max tailed users HR@1: {HIT_1_d2_tailed:.4f}/{best_hit_1_d2_tailed:.4f} \n, '
                        f'tailed users HR@5: {HIT_5_d2_tailed:.4f}/{best_hit_5_d2_tailed:.4f} \n, '
                        f'tailed users HR@10: {HIT_10_d2_tailed:.4f}/{best_hit_10_d2_tailed:.4f} \n'
                        f'tailed users NDCG@5: {NDCG_5_d2_tailed:.4f}/{best_ndcg_5_d2_tailed:.4f} \n, '
                        f'tailed users NDCG@10: {NDCG_10_d2_tailed:.4f}/{best_ndcg_10_d2_tailed:.4f}, \n'
                        f'tailed users MRR: {MRR_d2_tailed:.4f}/{best_mrr_d2_tailed:.4f} \n'
                        f'val domain2 cur/max cold-started users HR@1: {HIT_1_d2_cs:.4f}/{best_hit_1_d2_cs:.4f} \n, '
                        f'cold-started users HR@5: {HIT_5_d2_cs:.4f}/{best_hit_5_d2_cs:.4f} \n, '
                        f'cold-started users HR@10: {HIT_10_d2_cs:.4f}/{best_hit_10_d2_cs:.4f} \n'
                        f'cold-started users NDCG@5: {NDCG_5_d2_cs:.4f}/{best_ndcg_5_d2_cs:.4f} \n, '
                        f'cold-started users NDCG@10: {NDCG_10_d2_cs:.4f}/{best_ndcg_10_d2_cs:.4f}, \n'
                        f'cold-started users MRR: {MRR_d2_cs:.4f}/{best_mrr_d2_cs:.4f} \n'
                        f'val domain1 cur/max HR@1: {HIT_1_d1:.4f}/{best_hit_1_d1:.4f} \n,' 
                        f'HR@5: {HIT_5_d1:.4f}/{best_hit_5_d1:.4f} \n, '
                        f'HR@10: {HIT_10_d1:.4f}/{best_hit_10_d1:.4f} \n'
                        # f'val domain1 cur/max NDCG@1: {NDCG_1_d1:.4f}/{best_ndcg_1_d1:.4f} \n, '
                        f'NDCG@5: {NDCG_5_d1:.4f}/{best_ndcg_5_d1:.4f} \n, '
                        f'NDCG@10: {NDCG_10_d1:.4f}/{best_ndcg_10_d1:.4f}, \n'
                        f'val domain1 cur/max MRR: {MRR_d1:.4f}/{best_mrr_d1:.4f} \n'
                        f'val domain2 cur/max HR@1: {HIT_1_d2:.4f}/{best_hit_1_d2:.4f} \n, '
                        f'HR@5: {HIT_5_d2:.4f}/{best_hit_5_d2:.4f} \n, '
                        f'HR@10: {HIT_10_d2:.4f}/{best_hit_10_d2:.4f} \n'
                        # f'val domain2 cur/max NDCG@1: {NDCG_1_d2:.4f}/{best_ndcg_1_d2:.4f} \n, '
                        f'NDCG@5: {NDCG_5_d2:.4f}/{best_ndcg_5_d2:.4f} \n, '
                        f'NDCG@10: {NDCG_10_d2:.4f}/{best_ndcg_10_d2:.4f}, \n'
                        f'val domain2 cur/max MRR: {MRR_d2:.4f}/{best_mrr_d2:.4f} \n')
        else:
            val_loss, val_cls_loss, HIT_1_d1, NDCG_1_d1, HIT_5_d1, NDCG_5_d1, HIT_10_d1, NDCG_10_d1, MRR_d1, HIT_1_d2, NDCG_1_d2, HIT_5_d2, NDCG_5_d2, HIT_10_d2, NDCG_10_d2, MRR_d2 = test(model,args,valLoader)
            best_hit_1_d1 = max(HIT_1_d1,best_hit_1_d1)
            best_hit_5_d1 = max(HIT_5_d1,best_hit_5_d1)
            best_hit_10_d1 = max(HIT_10_d1,best_hit_10_d1)
            best_hit_1_d2 = max(HIT_1_d2,best_hit_1_d2)
            best_hit_5_d2 = max(HIT_5_d2,best_hit_5_d2)
            best_hit_10_d2 = max(HIT_10_d2,best_hit_10_d2)

            best_ndcg_1_d1 = max(best_ndcg_1_d1,NDCG_1_d1)
            best_ndcg_5_d1 = max(best_ndcg_5_d1,NDCG_5_d1)
            best_ndcg_10_d1 = max(best_ndcg_10_d1,NDCG_10_d1)
            best_ndcg_1_d2 = max(best_ndcg_1_d2,NDCG_1_d2)
            best_ndcg_5_d2 = max(best_ndcg_5_d2,NDCG_5_d2)
            best_ndcg_10_d2 = max(best_ndcg_10_d2,NDCG_10_d2)
            # if MRR_d1 >= best_mrr_d1:
            #     best_auc1 = auc_testd1
                # torch.save(model.state_dict(), str(save_path1))
            # if MRR_d2 >= best_mrr_d2:
            #     best_auc2 = auc_testd2
                # torch.save(model.state_dict(), str(save_path2))
            best_mrr_d1 = max(best_mrr_d1,MRR_d1)
            best_mrr_d2 = max(best_mrr_d2,MRR_d2)       
            logger.info(f'Epoch: {epoch}/{args.epoch} \t'
                        f'Train Loss: {stats.loss:.4f} \t'
                        f'Val loss: {val_loss:.4f}, cls loss: {val_cls_loss:.4f}\n'
                        f'val domain1 cur/max HR@1: {HIT_1_d1:.4f}/{best_hit_1_d1:.4f} \n,' 
                        f'HR@5: {HIT_5_d1:.4f}/{best_hit_5_d1:.4f} \n, '
                        f'HR@10: {HIT_10_d1:.4f}/{best_hit_10_d1:.4f} \n'
                        # f'val domain1 cur/max NDCG@1: {NDCG_1_d1:.4f}/{best_ndcg_1_d1:.4f} \n, '
                        f'NDCG@5: {NDCG_5_d1:.4f}/{best_ndcg_5_d1:.4f} \n, '
                        f'NDCG@10: {NDCG_10_d1:.4f}/{best_ndcg_10_d1:.4f}, \n'
                        f'val domain1 cur/max MRR: {MRR_d1:.4f}/{best_mrr_d1:.4f} \n'
                        f'val domain2 cur/max HR@1: {HIT_1_d2:.4f}/{best_hit_1_d2:.4f} \n, '
                        f'HR@5: {HIT_5_d2:.4f}/{best_hit_5_d2:.4f} \n, '
                        f'HR@10: {HIT_10_d2:.4f}/{best_hit_10_d2:.4f} \n'
                        # f'val domain2 cur/max NDCG@1: {NDCG_1_d2:.4f}/{best_ndcg_1_d2:.4f} \n, '
                        f'NDCG@5: {NDCG_5_d2:.4f}/{best_ndcg_5_d2:.4f} \n, '
                        f'NDCG@10: {NDCG_10_d2:.4f}/{best_ndcg_10_d2:.4f}, \n'
                        f'val domain2 cur/max MRR: {MRR_d2:.4f}/{best_mrr_d2:.4f} \n')
    if not args.overlap:     
        return best_hit_1_d1, best_hit_5_d1, best_hit_10_d1, best_ndcg_5_d1, best_ndcg_10_d1, best_mrr_d1, best_hit_1_d2, best_hit_5_d2, best_hit_10_d2, best_ndcg_5_d2, best_ndcg_10_d2, best_mrr_d2 
    else:
        return best_hit_1_d1_tailed, best_hit_5_d1_tailed, best_hit_10_d1_tailed, best_ndcg_5_d1_tailed, best_ndcg_10_d1_tailed, best_mrr_d1_tailed, best_hit_1_d1_cs, best_hit_5_d1_cs, best_hit_10_d1_cs, best_ndcg_5_d1_cs, best_ndcg_10_d1_cs, best_mrr_d1_cs, best_hit_1_d2_tailed, best_hit_5_d2_tailed, best_hit_10_d2_tailed, best_ndcg_5_d2_tailed, best_ndcg_10_d2_tailed, best_mrr_d2_tailed, best_hit_1_d2_cs, best_hit_5_d2_cs, best_hit_10_d2_cs, best_ndcg_5_d2_cs, best_ndcg_10_d2_cs, best_mrr_d2_cs, best_hit_1_d1, best_hit_5_d1, best_hit_10_d1, best_ndcg_5_d1, best_ndcg_10_d1, best_mrr_d1, best_hit_1_d2, best_hit_5_d2, best_hit_10_d2, best_ndcg_5_d2, best_ndcg_10_d2, best_mrr_d2  

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Multi-edge multi-domain training')
    parser.add_argument('--epoch', type=int, default=50, help='# of epoch')
    parser.add_argument('--bs', type=int, default=256, help='# images in batch')
    parser.add_argument('--use_gpu', type=bool, default=True, help='gpu flag, true for GPU and false for CPU')
    parser.add_argument('--lr', type=float, default=5e-4, help='initial learning rate for adam') #1e-3 for cdr23 3e-4 for cdr12
    parser.add_argument('--emb_dim', type=int, default=128, help='embedding size')
    parser.add_argument('--hid_dim', type=int, default=32, help='hidden layer dim')
    parser.add_argument('--seq_len', type=int, default=60, help='the length of the sequence') # 20 for mybank 150 for amazon
    parser.add_argument('--view_len', type=int, default=100, help='the length of the sequence') # 20 for mybank 150 for amazon
    parser.add_argument('--graph_nums', type=int, default=2, help='numbers of graph layers')
    parser.add_argument('--head_nums', type=int, default=32, help='head nums for u-u graph')
    parser.add_argument('--long_length', type=int, default=7, help='the length for setting long-tail node')
    parser.add_argument('--m1_layers', type=int, default=3, help='m1 layer nums')
    parser.add_argument('--m2_layers', type=int, default=3, help='m2 layer nums')
    parser.add_argument('--m3_layers', type=int, default=4, help='m3 layer nums')
    parser.add_argument('--m4_layers', type=int, default=2, help='m4 layer nums')
    parser.add_argument('--alpha_l', type=int, default=3, help='sce loss')
    parser.add_argument('--neg_nums', type=int, default=199, help='sample negative numbers')
    parser.add_argument('--mask_rate_enc', type=float, default=0.9, help='mask rate for encoder')
    parser.add_argument('--mask_rate_dec', type=float, default=0.9, help='mask rate for decoder')
    parser.add_argument('--overlap_ratio', type=float, default=0.25, help='overlap ratio for choose dataset ')
    parser.add_argument('--density_ratio', type=float, default=0.25, help='density ratio for choose dataset ')
    parser.add_argument('--cs_ratio', type=float, default=0.25, help='density ratio for choose dataset ')
    parser.add_argument('--bs_ratio', type=float, default=0.5, help='user-user connect ratio in the mini-batch graph')
    parser.add_argument('-md','--model-dir', type=str, default='model/')
    parser.add_argument('--log-file', type=str, default='log')
    parser.add_argument('--model', type=str, default='model select')
    parser.add_argument('-ds','--dataset_type', type=str, default='amazon')
    parser.add_argument('-dm','--domain_type', type=str, default='music_movie')
    parser.add_argument('--isInC', type=bool, default=False, help='add inc ')
    parser.add_argument('--isItC', type=bool, default=False, help='add itc')    
    parser.add_argument('--ts1', type=float, default=0.5, help='mask rate for encoder')
    parser.add_argument('--ts2', type=float, default=0.5, help='mask rate for decoder')
    parser.add_argument('--isCL', type=bool, default=False, help='add inc')
    parser.add_argument('--CL_w', type=float, default=0.1, help='add inc ')
    parser.add_argument('--overlap', type=bool, default=True, help='divided the performance by the overlapped users and non-overlapped users')    


    args = parser.parse_args()

    hit_1_d1 = []
    hit_5_d1 = []
    hit_10_d1 = []
    hit_1_d2 = []
    hit_5_d2 = []
    hit_10_d2 = []

    ndcg_5_d1 = []
    ndcg_10_d1 = []
    ndcg_5_d2 = []
    ndcg_10_d2 = []

    mrr_d1 = []
    mrr_d2 = []

    hit_1_d1_tailed = []
    hit_5_d1_tailed = []
    hit_10_d1_tailed = []
    hit_1_d2_tailed = []
    hit_5_d2_tailed = []
    hit_10_d2_tailed = []

    ndcg_5_d1_tailed = []
    ndcg_10_d1_tailed = []
    ndcg_5_d2_tailed = []
    ndcg_10_d2_tailed = []

    mrr_d1_tailed = []
    mrr_d2_tailed = []

    hit_1_d1_cs = []
    hit_5_d1_cs = []
    hit_10_d1_cs = []
    hit_1_d2_cs = []
    hit_5_d2_cs = []
    hit_10_d2_cs = []

    ndcg_5_d1_cs = []
    ndcg_10_d1_cs = []
    ndcg_5_d2_cs = []
    ndcg_10_d2_cs = []

    mrr_d1_cs = []
    mrr_d2_cs = []

    for i in range(5):
        SEED = i
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        args.log_file = "log" + str(i) + ".txt"
        user_length = 30626
        if args.domain_type == "music_movie":
            item_length = 97000
        elif args.domain_type == "book_movie":
            item_length = 444097
        elif args.domain_type == "cloth_sport":
            item_length = 24236
        elif args.domain_type == "phone_elec":
            item_length = 37703
        # item_length = 97000# 97000 music_movie #book_moive 444097
        datasetTrain = DualDomainSeqDataset(seq_len=args.seq_len,view_len=args.view_len,isTrain=True,neg_nums=args.neg_nums,long_length=args.long_length,pad_id=item_length,csv_path="./dataset/{}_train".format(args.domain_type)+str(int(args.overlap_ratio*100))+".csv")
        trainLoader = data.DataLoader(datasetTrain, batch_size=args.bs, shuffle=True, num_workers=8,drop_last=True,collate_fn=collate_fn_enhance)

        datasetVal = DualDomainSeqDataset(seq_len=args.seq_len,view_len=args.view_len,isTrain=False,neg_nums=args.neg_nums,long_length=args.long_length,pad_id=item_length,csv_path="./dataset/{}_test_csu".format(args.domain_type)+".csv")
        valLoader = data.DataLoader(datasetVal, batch_size=args.bs, shuffle=False, num_workers=8,drop_last=True,collate_fn=collate_fn_enhance)
        item_length = item_length * 2  #for pad id
        user_length = user_length * 2
        if args.model.lower() == "gru4rec":
            model = GRU4RecCDI(user_length=user_length, user_emb_dim=args.emb_dim, item_length=item_length, item_emb_dim=args.emb_dim, seq_len=args.seq_len, view_len=args.view_len, hid_dim=args.hid_dim, bs=args.bs, isCL=args.isCL).cuda()
        elif args.model.lower() == "sasrec":
            model = SASRecCDI(user_length=user_length, user_emb_dim=args.emb_dim, item_length=item_length, item_emb_dim=args.emb_dim, seq_len=args.seq_len, view_len=args.view_len, hid_dim=args.hid_dim, bs=args.bs, isCL=args.isCL).cuda()
        elif args.model.lower() == "bert4rec":
            model = BERT4RecCDI(user_length=user_length, user_emb_dim=args.emb_dim, item_length=item_length, item_emb_dim=args.emb_dim, seq_len=args.seq_len, view_len=args.view_len, hid_dim=args.hid_dim, bs=args.bs, isCL=args.isCL).cuda()
        cuda = True if torch.cuda.is_available() else False
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("find cuda right !!\n")
        # if cuda:
        #     torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # else:
        #     torch.set_default_tensor_type('torch.FloatTensor')

        if cuda:
            #model = torch.nn.DataParallel(model)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            #cudnn.benchmark = True
            model = model.cuda()
            # model.to(device)
            print("use cuda!")
        optimizer = torch.optim.Adam(model.parameters(),lr = args.lr)
        init_logger(args.model_dir, args.log_file)
        logger.info(vars(args))
        # if os.path.exists(args.model_dir + "best_d1.pt"):
        #     print("load_pretrained")
        #     state_dict = torch.load(args.model_dir + "best_d1.pt")
        #     model.load_state_dict(state_dict,strict=False)
        if not args.overlap:
            best_hit_1_d1, best_hit_5_d1, best_hit_10_d1, best_ndcg_5_d1, best_ndcg_10_d1, best_mrr_d1, best_hit_1_d2, best_hit_5_d2, best_hit_10_d2, best_ndcg_5_d2, best_ndcg_10_d2, best_mrr_d2  = train(model,trainLoader,args,valLoader)
            # test(model,args,valLoader)
            hit_1_d1.append(best_hit_1_d1)
            hit_5_d1.append(best_hit_5_d1)
            hit_10_d1.append(best_hit_10_d1)
            ndcg_5_d1.append(best_ndcg_5_d1)
            ndcg_10_d1.append(best_ndcg_10_d1)
            mrr_d1.append(best_mrr_d1)

            hit_1_d2.append(best_hit_1_d2)
            hit_5_d2.append(best_hit_5_d2)
            hit_10_d2.append(best_hit_10_d2)
            ndcg_5_d2.append(best_ndcg_5_d2)
            ndcg_10_d2.append(best_ndcg_10_d2)
            mrr_d2.append(best_mrr_d2)
            # break
        else:
            best_hit_1_d1_tailed, best_hit_5_d1_tailed, best_hit_10_d1_tailed, best_ndcg_5_d1_tailed, best_ndcg_10_d1_tailed, best_mrr_d1_tailed, best_hit_1_d1_cs, best_hit_5_d1_cs, best_hit_10_d1_cs, best_ndcg_5_d1_cs, best_ndcg_10_d1_cs, best_mrr_d1_cs, best_hit_1_d2_tailed, best_hit_5_d2_tailed, best_hit_10_d2_tailed, best_ndcg_5_d2_tailed, best_ndcg_10_d2_tailed, best_mrr_d2_tailed, best_hit_1_d2_cs, best_hit_5_d2_cs, best_hit_10_d2_cs, best_ndcg_5_d2_cs, best_ndcg_10_d2_cs, best_mrr_d2_cs, best_hit_1_d1, best_hit_5_d1, best_hit_10_d1, best_ndcg_5_d1, best_ndcg_10_d1, best_mrr_d1, best_hit_1_d2, best_hit_5_d2, best_hit_10_d2, best_ndcg_5_d2, best_ndcg_10_d2, best_mrr_d2  = train(model,trainLoader,args,valLoader)
            # test(model,args,valLoader)
            hit_1_d1_tailed.append(best_hit_1_d1_tailed)
            hit_5_d1_tailed.append(best_hit_5_d1_tailed)
            hit_10_d1_tailed.append(best_hit_10_d1_tailed)
            ndcg_5_d1_tailed.append(best_ndcg_5_d1_tailed)
            ndcg_10_d1_tailed.append(best_ndcg_10_d1_tailed)
            mrr_d1_tailed.append(best_mrr_d1_tailed)

            hit_1_d2_tailed.append(best_hit_1_d2_tailed)
            hit_5_d2_tailed.append(best_hit_5_d2_tailed)
            hit_10_d2_tailed.append(best_hit_10_d2_tailed)
            ndcg_5_d2_tailed.append(best_ndcg_5_d2_tailed)
            ndcg_10_d2_tailed.append(best_ndcg_10_d2_tailed)
            mrr_d2_tailed.append(best_mrr_d2_tailed)

            hit_1_d1_cs.append(best_hit_1_d1_cs)
            hit_5_d1_cs.append(best_hit_5_d1_cs)
            hit_10_d1_cs.append(best_hit_10_d1_cs)
            ndcg_5_d1_cs.append(best_ndcg_5_d1_cs)
            ndcg_10_d1_cs.append(best_ndcg_10_d1_cs)
            mrr_d1_cs.append(best_mrr_d1_cs)

            hit_1_d2_cs.append(best_hit_1_d2_cs)
            hit_5_d2_cs.append(best_hit_5_d2_cs)
            hit_10_d2_cs.append(best_hit_10_d2_cs)
            ndcg_5_d2_cs.append(best_ndcg_5_d2_cs)
            ndcg_10_d2_cs.append(best_ndcg_10_d2_cs)
            mrr_d2_cs.append(best_mrr_d2_cs)

            hit_1_d1.append(best_hit_1_d1)
            hit_5_d1.append(best_hit_5_d1)
            hit_10_d1.append(best_hit_10_d1)
            ndcg_5_d1.append(best_ndcg_5_d1)
            ndcg_10_d1.append(best_ndcg_10_d1)
            mrr_d1.append(best_mrr_d1)

            hit_1_d2.append(best_hit_1_d2)
            hit_5_d2.append(best_hit_5_d2)
            hit_10_d2.append(best_hit_10_d2)
            ndcg_5_d2.append(best_ndcg_5_d2)
            ndcg_10_d2.append(best_ndcg_10_d2)
            mrr_d2.append(best_mrr_d2)

    if not args.overlap:
        log_all_txt = "log_all.txt"
        init_logger(args.model_dir, log_all_txt)
        logger.info(f'domain1 HR@1: {np.mean(hit_1_d1):.4f}/{np.std(hit_1_d1):.4f} \n,' 
                    f'HR@5: {np.mean(hit_5_d1):.4f}/{np.std(hit_5_d1):.4f} \n, '
                    f'HR@10: {np.mean(hit_10_d1):.4f}/{np.std(hit_10_d1):.4f} \n'
                    f'NDCG@5: {np.mean(ndcg_5_d1):.4f}/{np.std(ndcg_5_d1):.4f} \n, '
                    f'NDCG@10: {np.mean(ndcg_10_d1):.4f}/{np.std(ndcg_10_d1):.4f}, \n'
                    f'MRR: {np.mean(mrr_d1):.4f}/{np.std(mrr_d1):.4f} \n'
                    f'domain2 HR@1: {np.mean(hit_1_d2):.4f}/{np.std(hit_1_d2):.4f} \n,' 
                    f'HR@5: {np.mean(hit_5_d2):.4f}/{np.std(hit_5_d2):.4f} \n, '
                    f'HR@10: {np.mean(hit_10_d2):.4f}/{np.std(hit_10_d2):.4f} \n'
                    f'NDCG@5: {np.mean(ndcg_5_d2):.4f}/{np.std(ndcg_5_d2):.4f} \n, '
                    f'NDCG@10: {np.mean(ndcg_10_d2):.4f}/{np.std(ndcg_10_d2):.4f}, \n'
                    f'MRR: {np.mean(mrr_d2):.4f}/{np.std(mrr_d2):.4f} \n'
                    f'Avg HR@1: {(np.mean(hit_1_d2)+np.mean(hit_1_d1))/2:.4f}/{(np.std(hit_1_d2)**2+np.std(hit_1_d1)**2)**0.5:.4f} \n,' 
                    f'HR@5: {(np.mean(hit_5_d2)+np.mean(hit_5_d1))/2:.4f}/{(np.std(hit_5_d2)**2+np.std(hit_5_d1)**2)**0.5:.4f} \n, '
                    f'HR@10: {(np.mean(hit_10_d2)+np.mean(hit_10_d1))/2:.4f}/{(np.std(hit_10_d2)**2+np.std(hit_10_d1)**2)**0.5:.4f} \n'
                    f'NDCG@5: {(np.mean(ndcg_5_d2)+np.mean(ndcg_5_d1))/2:.4f}/{(np.std(ndcg_5_d2)**2+np.std(ndcg_5_d1)**2)**0.5:.4f} \n, '
                    f'NDCG@10: {(np.mean(ndcg_10_d2)+np.mean(ndcg_10_d1))/2:.4f}/{(np.std(ndcg_10_d2)**2+np.std(ndcg_10_d1)**2)**0.5:.4f}, \n'
                    f'MRR: {(np.mean(mrr_d2)+np.mean(mrr_d1))/2:.4f}/{(np.std(mrr_d2)**2+np.std(mrr_d1)**2)**0.5:.4f} \n')
    else:
        log_all_txt = "log_all.txt"
        init_logger(args.model_dir, log_all_txt)
        logger.info(f'domain1 tailed users HR@1: {np.mean(hit_1_d1_tailed):.4f}/{np.std(hit_1_d1_tailed):.4f} \n,' 
                    f'tailed users HR@5: {np.mean(hit_5_d1_tailed):.4f}/{np.std(hit_5_d1_tailed):.4f} \n, '
                    f'tailed users HR@10: {np.mean(hit_10_d1_tailed):.4f}/{np.std(hit_10_d1_tailed):.4f} \n'
                    f'tailed users NDCG@5: {np.mean(ndcg_5_d1_tailed):.4f}/{np.std(ndcg_5_d1_tailed):.4f} \n, '
                    f'tailed users NDCG@10: {np.mean(ndcg_10_d1_tailed):.4f}/{np.std(ndcg_10_d1_tailed):.4f}, \n'
                    f'tailed users MRR: {np.mean(mrr_d1_tailed):.4f}/{np.std(mrr_d1_tailed):.4f} \n'
                    f'domain1 cold-started users HR@1: {np.mean(hit_1_d1_cs):.4f}/{np.std(hit_1_d1_cs):.4f} \n,' 
                    f'cold-started users HR@5: {np.mean(hit_5_d1_cs):.4f}/{np.std(hit_5_d1_cs):.4f} \n, '
                    f'cold-started users HR@10: {np.mean(hit_10_d1_cs):.4f}/{np.std(hit_10_d1_cs):.4f} \n'
                    f'cold-started users NDCG@5: {np.mean(ndcg_5_d1_cs):.4f}/{np.std(ndcg_5_d1_cs):.4f} \n, '
                    f'cold-started users NDCG@10: {np.mean(ndcg_10_d1_cs):.4f}/{np.std(ndcg_10_d1_cs):.4f}, \n'
                    f'cold-started users MRR: {np.mean(mrr_d1_cs):.4f}/{np.std(mrr_d1_cs):.4f} \n'
                    f'tailed users domain2 HR@1: {np.mean(hit_1_d2_tailed):.4f}/{np.std(hit_1_d2_tailed):.4f} \n,' 
                    f'tailed users HR@5: {np.mean(hit_5_d2_tailed):.4f}/{np.std(hit_5_d2_tailed):.4f} \n, '
                    f'tailed users HR@10: {np.mean(hit_10_d2_tailed):.4f}/{np.std(hit_10_d2_tailed):.4f} \n'
                    f'tailed users NDCG@5: {np.mean(ndcg_5_d2_tailed):.4f}/{np.std(ndcg_5_d2_tailed):.4f} \n, '
                    f'tailed users NDCG@10: {np.mean(ndcg_10_d2_tailed):.4f}/{np.std(ndcg_10_d2_tailed):.4f}, \n'
                    f'tailed users MRR: {np.mean(mrr_d2_tailed):.4f}/{np.std(mrr_d2_tailed):.4f} \n'
                    f'cold-started users domain2 HR@1: {np.mean(hit_1_d2_cs):.4f}/{np.std(hit_1_d2_cs):.4f} \n,' 
                    f'cold-started users HR@5: {np.mean(hit_5_d2_cs):.4f}/{np.std(hit_5_d2_cs):.4f} \n, '
                    f'cold-started users HR@10: {np.mean(hit_10_d2_cs):.4f}/{np.std(hit_10_d2_cs):.4f} \n'
                    f'cold-started users NDCG@5: {np.mean(ndcg_5_d2_cs):.4f}/{np.std(ndcg_5_d2_cs):.4f} \n, '
                    f'cold-started users NDCG@10: {np.mean(ndcg_10_d2_cs):.4f}/{np.std(ndcg_10_d2_cs):.4f}, \n'
                    f'cold-started users MRR: {np.mean(mrr_d2_cs):.4f}/{np.std(mrr_d2_cs):.4f} \n'
                    f'tailed users Avg HR@1: {(np.mean(hit_1_d2_tailed)+np.mean(hit_1_d1_tailed))/2:.4f}/{(np.std(hit_1_d2_tailed)**2+np.std(hit_1_d1_tailed)**2)**0.5:.4f} \n,' 
                    f'tailed users HR@5: {(np.mean(hit_5_d2_tailed)+np.mean(hit_5_d1_tailed))/2:.4f}/{(np.std(hit_5_d2_tailed)**2+np.std(hit_5_d1_tailed)**2)**0.5:.4f} \n, '
                    f'tailed users HR@10: {(np.mean(hit_10_d2_tailed)+np.mean(hit_10_d1_tailed))/2:.4f}/{(np.std(hit_10_d2_tailed)**2+np.std(hit_10_d1_tailed)**2)**0.5:.4f} \n'
                    f'tailed users NDCG@5: {(np.mean(ndcg_5_d2_tailed)+np.mean(ndcg_5_d1_tailed))/2:.4f}/{(np.std(ndcg_5_d2_tailed)**2+np.std(ndcg_5_d1_tailed)**2)**0.5:.4f} \n, '
                    f'tailed users NDCG@10: {(np.mean(ndcg_10_d2_tailed)+np.mean(ndcg_10_d1_tailed))/2:.4f}/{(np.std(ndcg_10_d2_tailed)**2+np.std(ndcg_10_d1_tailed)**2)**0.5:.4f}, \n'
                    f'tailed users MRR: {(np.mean(mrr_d2_tailed)+np.mean(mrr_d1_tailed))/2:.4f}/{(np.std(mrr_d2_tailed)**2+np.std(mrr_d1_tailed)**2)**0.5:.4f} \n'
                    f'cold-started users Avg HR@1: {(np.mean(hit_1_d2_cs)+np.mean(hit_1_d1_cs))/2:.4f}/{(np.std(hit_1_d2_cs)**2+np.std(hit_1_d1_cs)**2)**0.5:.4f} \n,' 
                    f'cold-started users HR@5: {(np.mean(hit_5_d2_cs)+np.mean(hit_5_d1_cs))/2:.4f}/{(np.std(hit_5_d2_cs)**2+np.std(hit_5_d1_cs)**2)**0.5:.4f} \n, '
                    f'cold-started users HR@10: {(np.mean(hit_10_d2_cs)+np.mean(hit_10_d1_cs))/2:.4f}/{(np.std(hit_10_d2_cs)**2+np.std(hit_10_d1_cs)**2)**0.5:.4f} \n'
                    f'cold-started users NDCG@5: {(np.mean(ndcg_5_d2_cs)+np.mean(ndcg_5_d1_cs))/2:.4f}/{(np.std(ndcg_5_d2_cs)**2+np.std(ndcg_5_d1_cs)**2)**0.5:.4f} \n, '
                    f'cold-started users NDCG@10: {(np.mean(ndcg_10_d2_cs)+np.mean(ndcg_10_d1_cs))/2:.4f}/{(np.std(ndcg_10_d2_cs)**2+np.std(ndcg_10_d1_cs)**2)**0.5:.4f}, \n'
                    f'cold-started users MRR: {(np.mean(mrr_d2_cs)+np.mean(mrr_d1_cs))/2:.4f}/{(np.std(mrr_d2_cs)**2+np.std(mrr_d1_cs)**2)**0.5:.4f} \n'
                    f'domain1 HR@1: {np.mean(hit_1_d1):.4f}/{np.std(hit_1_d1):.4f} \n,' 
                    f'HR@5: {np.mean(hit_5_d1):.4f}/{np.std(hit_5_d1):.4f} \n, '
                    f'HR@10: {np.mean(hit_10_d1):.4f}/{np.std(hit_10_d1):.4f} \n'
                    f'NDCG@5: {np.mean(ndcg_5_d1):.4f}/{np.std(ndcg_5_d1):.4f} \n, '
                    f'NDCG@10: {np.mean(ndcg_10_d1):.4f}/{np.std(ndcg_10_d1):.4f}, \n'
                    f'MRR: {np.mean(mrr_d1):.4f}/{np.std(mrr_d1):.4f} \n'
                    f'domain2 HR@1: {np.mean(hit_1_d2):.4f}/{np.std(hit_1_d2):.4f} \n,' 
                    f'HR@5: {np.mean(hit_5_d2):.4f}/{np.std(hit_5_d2):.4f} \n, '
                    f'HR@10: {np.mean(hit_10_d2):.4f}/{np.std(hit_10_d2):.4f} \n'
                    f'NDCG@5: {np.mean(ndcg_5_d2):.4f}/{np.std(ndcg_5_d2):.4f} \n, '
                    f'NDCG@10: {np.mean(ndcg_10_d2):.4f}/{np.std(ndcg_10_d2):.4f}, \n'
                    f'MRR: {np.mean(mrr_d2):.4f}/{np.std(mrr_d2):.4f} \n'
                    f'Avg HR@1: {(np.mean(hit_1_d2)+np.mean(hit_1_d1))/2:.4f}/{(np.std(hit_1_d2)**2+np.std(hit_1_d1)**2)**0.5:.4f} \n,' 
                    f'HR@5: {(np.mean(hit_5_d2)+np.mean(hit_5_d1))/2:.4f}/{(np.std(hit_5_d2)**2+np.std(hit_5_d1)**2)**0.5:.4f} \n, '
                    f'HR@10: {(np.mean(hit_10_d2)+np.mean(hit_10_d1))/2:.4f}/{(np.std(hit_10_d2)**2+np.std(hit_10_d1)**2)**0.5:.4f} \n'
                    f'NDCG@5: {(np.mean(ndcg_5_d2)+np.mean(ndcg_5_d1))/2:.4f}/{(np.std(ndcg_5_d2)**2+np.std(ndcg_5_d1)**2)**0.5:.4f} \n, '
                    f'NDCG@10: {(np.mean(ndcg_10_d2)+np.mean(ndcg_10_d1))/2:.4f}/{(np.std(ndcg_10_d2)**2+np.std(ndcg_10_d1)**2)**0.5:.4f}, \n'
                    f'MRR: {(np.mean(mrr_d2)+np.mean(mrr_d1))/2:.4f}/{(np.std(mrr_d2)**2+np.std(mrr_d1)**2)**0.5:.4f} \n')