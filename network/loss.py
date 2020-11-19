#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from .lovasz_losses import lovasz_softmax

def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        (https://github.com/tianweiy/CenterPoint)
    Arguments:
        pred (batch x c x h x w)
        gt (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    # loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    return - (pos_loss + neg_loss)

class FocalLoss(torch.nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)

class panoptic_loss(torch.nn.Module):
    def __init__(self, ignore_label = 255, center_loss_weight = 100, offset_loss_weight = 1, per_class_heatmap = False, center_loss = 'MSE', offset_loss = 'L1'):
        super(panoptic_loss, self).__init__()
        self.CE_loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)
        assert center_loss in ['MSE','FocalLoss']
        assert offset_loss in ['L1','SmoothL1']
        if center_loss == 'MSE':
            self.center_loss_fn = torch.nn.MSELoss()
        elif center_loss == 'FocalLoss':
            self.center_loss_fn = FocalLoss()
        else: raise NotImplementedError
        if offset_loss == 'L1':
            self.offset_loss_fn = torch.nn.L1Loss()
        elif offset_loss == 'SmoothL1':
            self.offset_loss_fn = torch.nn.SmoothL1Loss()
        else: raise NotImplementedError
        self.center_loss_weight = center_loss_weight
        self.offset_loss_weight = offset_loss_weight
        self.per_class_heatmap = per_class_heatmap

        print('Using '+ center_loss +' for heatmap regression, weight: '+str(center_loss_weight))
        print('Using '+ offset_loss +' for offset regression, weight: '+str(offset_loss_weight))

        self.lost_dict={'semantic_loss':[],
                        'heatmap_loss':[],
                        'offset_loss':[]}

    def reset_loss_dict(self):
        self.lost_dict={'semantic_loss':[],
                        'heatmap_loss':[],
                        'offset_loss':[]}

    def forward(self,prediction,center,offset,gt_label,gt_center,gt_offset,save_loss = True):
        # semantic loss
        loss = lovasz_softmax(torch.nn.functional.softmax(prediction), gt_label,ignore=255) + self.CE_loss(prediction,gt_label)
        if save_loss:
            self.lost_dict['semantic_loss'].append(loss.item())
        # center heatmap loss
        if self.per_class_heatmap:
            center_mask = (torch.max(gt_center,dim=1,keepdim=True)[0]>0) | (torch.min(torch.unsqueeze(gt_label, 1),dim=4)[0]<255)
            center_mask = center_mask.repeat(1,8,1,1)
        else:
            center_mask = (gt_center>0) | (torch.min(torch.unsqueeze(gt_label, 1),dim=4)[0]<255)
        center_loss = self.center_loss_fn(center,gt_center) * center_mask
        # safe division
        if center_mask.sum() > 0:
            center_loss = center_loss.sum() / center_mask.sum() * self.center_loss_weight
        else:
            center_loss = center_loss.sum() * 0
        if save_loss:
            self.lost_dict['heatmap_loss'].append(center_loss.item())
        loss += center_loss
        # offset loss
        offset_mask = gt_offset != 0
        offset_loss = self.offset_loss_fn(offset,gt_offset) * offset_mask
        # safe division
        if offset_mask.sum() > 0:
            offset_loss = offset_loss.sum() / offset_mask.sum() * self.offset_loss_weight
        else:
            offset_loss = offset_loss.sum() * 0
        if save_loss:
            self.lost_dict['offset_loss'].append(offset_loss.item())
        loss += offset_loss
        return loss