#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import sys
import numpy as np
import yaml
import torch
import torch.optim as optim
from tqdm import tqdm

from network.BEV_Unet import BEV_Unet
from network.ptBEV import ptBEVnet
from dataloader.dataset import collate_fn_BEV,SemKITTI,SemKITTI_label_name,spherical_dataset,voxel_dataset
from network.instance_post_processing import get_panoptic_segmentation
from network.loss import panoptic_loss
from utils.eval_pq import PanopticEval
from utils.configs import merge_configs
#ignore weird np warning
import warnings
warnings.filterwarnings("ignore")

def SemKITTI2train(label):
    if isinstance(label, list):
        return [SemKITTI2train_single(a) for a in label]
    else:
        return SemKITTI2train_single(label)

def SemKITTI2train_single(label):
    return label - 1 # uint8 trick

def load_pretrained_model(model,pretrained_model):
    model_dict = model.state_dict()
    pretrained_model = {k: v for k, v in pretrained_model.items() if k in model_dict}
    model_dict.update(pretrained_model) 
    model.load_state_dict(model_dict)
    return model

def main(args):
    data_path = args['dataset']['path']
    train_batch_size = args['model']['train_batch_size']
    val_batch_size = args['model']['val_batch_size']
    check_iter = args['model']['check_iter']
    model_save_path = args['model']['model_save_path']
    pretrained_model = args['model']['pretrained_model']
    compression_model = args['dataset']['grid_size'][2]
    grid_size = args['dataset']['grid_size']
    visibility = args['model']['visibility']
    pytorch_device = torch.device('cuda:0')
    if args['model']['polar']:
        fea_dim = 9
        circular_padding = True
    else:
        fea_dim = 7
        circular_padding = False

    #prepare miou fun
    unique_label=np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str=[SemKITTI_label_name[x] for x in unique_label+1]

    #prepare model
    my_BEV_model=BEV_Unet(n_class=len(unique_label), n_height = compression_model, input_batch_norm = True, dropout = 0.5, circular_padding = circular_padding, use_vis_fea=visibility)
    my_model = ptBEVnet(my_BEV_model, pt_model = 'pointnet', grid_size =  grid_size, fea_dim = fea_dim, max_pt_per_encode = 256,
                            out_pt_fea_dim = 512, kernal_size = 1, pt_selection = 'random', fea_compre = compression_model)
    if os.path.exists(model_save_path):
        my_model = load_pretrained_model(my_model,torch.load(model_save_path))
    elif os.path.exists(pretrained_model):
        my_model = load_pretrained_model(my_model,torch.load(pretrained_model))
    my_model.to(pytorch_device)

    optimizer = optim.Adam(my_model.parameters())
    loss_fn = panoptic_loss(center_loss_weight = args['model']['center_loss_weight'], offset_loss_weight = args['model']['offset_loss_weight'],\
                            center_loss = args['model']['center_loss'], offset_loss=args['model']['offset_loss'])

    #prepare dataset
    train_pt_dataset = SemKITTI(data_path + '/sequences/', imageset = 'train', return_ref = True, instance_pkl_path=args['dataset']['instance_pkl_path'])
    val_pt_dataset = SemKITTI(data_path + '/sequences/', imageset = 'val', return_ref = True, instance_pkl_path=args['dataset']['instance_pkl_path'])
    if args['model']['polar']:
        train_dataset=spherical_dataset(train_pt_dataset, args['dataset'], grid_size = grid_size, ignore_label = 0, use_aug = True)
        val_dataset=spherical_dataset(val_pt_dataset, args['dataset'], grid_size = grid_size, ignore_label = 0)
    else:
        train_dataset=voxel_dataset(train_pt_dataset, args['dataset'], grid_size = grid_size, ignore_label = 0,use_aug = True)
        val_dataset=voxel_dataset(val_pt_dataset, args['dataset'], grid_size = grid_size, ignore_label = 0)
    train_dataset_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                    batch_size = train_batch_size,
                                                    collate_fn = collate_fn_BEV,
                                                    shuffle = True,
                                                    num_workers = 4)
    val_dataset_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                                    batch_size = val_batch_size,
                                                    collate_fn = collate_fn_BEV,
                                                    shuffle = False,
                                                    num_workers = 4)

    # training
    epoch=0
    best_val_PQ=0
    start_training=False
    my_model.train()
    global_iter = 0
    exce_counter = 0
    evaluator = PanopticEval(len(unique_label)+1, None, [0], min_points=50)

    while epoch < args['model']['max_epoch']:
        pbar = tqdm(total=len(train_dataset_loader))
        for i_iter,(train_vox_fea,train_label_tensor,train_gt_center,train_gt_offset,train_grid,_,_,train_pt_fea) in enumerate(train_dataset_loader):
            # validation
            if global_iter % check_iter == 0:
                my_model.eval()
                evaluator.reset()
                with torch.no_grad():
                    for i_iter_val,(val_vox_fea,val_vox_label,val_gt_center,val_gt_offset,val_grid,val_pt_labels,val_pt_ints,val_pt_fea) in enumerate(val_dataset_loader):
                        val_vox_fea_ten = val_vox_fea.to(pytorch_device)
                        val_vox_label = SemKITTI2train(val_vox_label)
                        val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in val_pt_fea]
                        val_grid_ten = [torch.from_numpy(i[:,:2]).to(pytorch_device) for i in val_grid]
                        val_label_tensor=val_vox_label.type(torch.LongTensor).to(pytorch_device)
                        val_gt_center_tensor = val_gt_center.to(pytorch_device)
                        val_gt_offset_tensor = val_gt_offset.to(pytorch_device)

                        if visibility:
                            predict_labels,center,offset = my_model(val_pt_fea_ten, val_grid_ten, val_vox_fea_ten)
                        else:
                            predict_labels,center,offset = my_model(val_pt_fea_ten, val_grid_ten)

                        for count,i_val_grid in enumerate(val_grid):
                            # get foreground_mask
                            for_mask = torch.zeros(1,grid_size[0],grid_size[1],grid_size[2],dtype=torch.bool).to(pytorch_device)
                            for_mask[0,val_grid[count][:,0],val_grid[count][:,1],val_grid[count][:,2]] = True
                            # post processing
                            panoptic_labels,center_points = get_panoptic_segmentation(torch.unsqueeze(predict_labels[count], 0),torch.unsqueeze(center[count], 0),torch.unsqueeze(offset[count], 0),\
                                                                                      val_pt_dataset.thing_list, threshold=args['model']['post_proc']['threshold'], nms_kernel=args['model']['post_proc']['nms_kernel'],\
                                                                                      top_k=args['model']['post_proc']['top_k'], polar=circular_padding,foreground_mask=for_mask)
                            panoptic_labels = panoptic_labels.cpu().detach().numpy().astype(np.int32)
                            panoptic = panoptic_labels[0,val_grid[count][:,0],val_grid[count][:,1],val_grid[count][:,2]]
                            evaluator.addBatch(panoptic & 0xFFFF,panoptic,np.squeeze(val_pt_labels[count]),np.squeeze(val_pt_ints[count]))
                        del val_vox_label,val_pt_fea_ten,val_label_tensor,val_grid_ten,val_gt_center,val_gt_center_tensor,val_gt_offset,val_gt_offset_tensor,predict_labels,center,offset,panoptic_labels,center_points
                my_model.train()
                class_PQ, class_SQ, class_RQ, class_all_PQ, class_all_SQ, class_all_RQ = evaluator.getPQ()
                miou,ious = evaluator.getSemIoU()
                print('Validation per class PQ, SQ, RQ and IoU: ')
                for class_name, class_pq, class_sq, class_rq, class_iou in zip(unique_label_str,class_all_PQ[1:],class_all_SQ[1:],class_all_RQ[1:],ious[1:]):
                    print('%15s : %6.2f%%  %6.2f%%  %6.2f%%  %6.2f%%' % (class_name, class_pq*100, class_sq*100, class_rq*100, class_iou*100))                                  
                # save model if performance is improved
                if best_val_PQ<class_PQ:
                    best_val_PQ=class_PQ
                    torch.save(my_model.state_dict(), model_save_path)
                print('Current val PQ is %.3f while the best val PQ is %.3f' %
                    (class_PQ*100,best_val_PQ*100))               
                print('Current val miou is %.3f'%
                    (miou*100))

                if start_training:
                    sem_l ,hm_l, os_l = np.mean(loss_fn.lost_dict['semantic_loss']), np.mean(loss_fn.lost_dict['heatmap_loss']), np.mean(loss_fn.lost_dict['offset_loss'])
                    print('epoch %d iter %5d, loss: %.3f, semantic loss: %.3f, heatmap loss: %.3f, offset loss: %.3f\n' %
                        (epoch, i_iter, sem_l+hm_l+os_l, sem_l, hm_l, os_l))
                print('%d exceptions encountered during last training\n' %
                    exce_counter)
                exce_counter = 0
                loss_fn.reset_loss_dict()

            # training
            try:
                train_vox_fea_ten = train_vox_fea.to(pytorch_device)
                train_label_tensor = SemKITTI2train(train_label_tensor)
                train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in train_pt_fea]
                train_grid_ten = [torch.from_numpy(i[:,:2]).to(pytorch_device) for i in train_grid]
                train_label_tensor=train_label_tensor.type(torch.LongTensor).to(pytorch_device)
                train_gt_center_tensor = train_gt_center.to(pytorch_device)
                train_gt_offset_tensor = train_gt_offset.to(pytorch_device)

                if args['model']['enable_SAP'] and epoch>=args['model']['SAP']['start_epoch']:
                    for fea in train_pt_fea_ten:
                        fea.requires_grad_()
        
                # forward
                if visibility:
                    sem_prediction,center,offset = my_model(train_pt_fea_ten,train_grid_ten,train_vox_fea_ten)
                else:
                    sem_prediction,center,offset = my_model(train_pt_fea_ten,train_grid_ten)
                # loss
                loss = loss_fn(sem_prediction,center,offset,train_label_tensor,train_gt_center_tensor,train_gt_offset_tensor)
                
                
                # self adversarial pruning
                if args['model']['enable_SAP'] and epoch>=args['model']['SAP']['start_epoch']:
                    loss.backward()
                    for i,fea in enumerate(train_pt_fea_ten):
                        fea_grad = torch.norm(fea.grad,dim=1)
                        top_k_grad, _ = torch.topk(fea_grad, int(args['model']['SAP']['rate']*fea_grad.shape[0]))
                        # delete high influential points
                        train_pt_fea_ten[i] = train_pt_fea_ten[i][fea_grad < top_k_grad[-1]]
                        train_grid_ten[i] = train_grid_ten[i][fea_grad < top_k_grad[-1]]
                    optimizer.zero_grad()

                    # second pass
                    # forward
                    if visibility:
                        sem_prediction,center,offset = my_model(train_pt_fea_ten,train_grid_ten,train_vox_fea_ten)
                    else:
                        sem_prediction,center,offset = my_model(train_pt_fea_ten,train_grid_ten)
                    # loss
                    loss = loss_fn(sem_prediction,center,offset,train_label_tensor,train_gt_center_tensor,train_gt_offset_tensor)
                    
                # backward + optimize
                loss.backward()
                optimizer.step()
            except Exception as error: 
                if exce_counter == 0:
                    print(error)
                exce_counter += 1
            
            # zero the parameter gradients
            optimizer.zero_grad()
            pbar.update(1)
            start_training=True
            global_iter += 1
        pbar.close()
        epoch += 1

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--data_dir', default='data')
    parser.add_argument('-p', '--model_save_path', default='./Panoptic_SemKITTI.pt')
    parser.add_argument('-c', '--configs', default='configs/SemanticKITTI_model/Panoptic-PolarNet.yaml')
    parser.add_argument('--pretrained_model', default='empty')

    args = parser.parse_args()
    with open(args.configs, 'r') as s:
        new_args = yaml.safe_load(s)
    args = merge_configs(args,new_args)

    print(' '.join(sys.argv))
    print(args)
    main(args)