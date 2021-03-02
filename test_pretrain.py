#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import argparse
import sys
import yaml
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import errno

from network.BEV_Unet import BEV_Unet
from network.ptBEV import ptBEVnet
from dataloader.dataset import collate_fn_BEV,SemKITTI,SemKITTI_label_name,spherical_dataset,voxel_dataset,collate_fn_BEV_test
from network.instance_post_processing import get_panoptic_segmentation
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

def main(args):
    data_path = args['dataset']['path']
    test_batch_size = args['model']['test_batch_size']
    pretrained_model = args['model']['pretrained_model']
    output_path = args['dataset']['output_path']
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

    # prepare miou fun
    unique_label=np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str=[SemKITTI_label_name[x] for x in unique_label+1]

    # prepare model
    my_BEV_model=BEV_Unet(n_class=len(unique_label), n_height = compression_model, input_batch_norm = True, dropout = 0.5, circular_padding = circular_padding, use_vis_fea=visibility)
    my_model = ptBEVnet(my_BEV_model, pt_model = 'pointnet', grid_size =  grid_size, fea_dim = fea_dim, max_pt_per_encode = 256,
                            out_pt_fea_dim = 512, kernal_size = 1, pt_selection = 'random', fea_compre = compression_model)
    if os.path.exists(pretrained_model):
        my_model.load_state_dict(torch.load(pretrained_model))
    pytorch_total_params = sum(p.numel() for p in my_model.parameters())
    print('params: ',pytorch_total_params)
    my_model.to(pytorch_device)
    my_model.eval()

    # prepare dataset
    test_pt_dataset = SemKITTI(data_path + '/sequences/', imageset = 'test', return_ref = True, instance_pkl_path=args['dataset']['instance_pkl_path'])
    val_pt_dataset = SemKITTI(data_path + '/sequences/', imageset = 'val', return_ref = True, instance_pkl_path=args['dataset']['instance_pkl_path'])
    if args['model']['polar']:
        test_dataset=spherical_dataset(test_pt_dataset, args['dataset'], grid_size = grid_size, ignore_label = 0, return_test= True)
        val_dataset=spherical_dataset(val_pt_dataset, args['dataset'], grid_size = grid_size, ignore_label = 0)
    else:
        test_dataset=voxel_dataset(test_pt_dataset, args['dataset'], grid_size = grid_size, ignore_label = 0, return_test= True)
        val_dataset=voxel_dataset(val_pt_dataset, args['dataset'], grid_size = grid_size, ignore_label = 0)
    test_dataset_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                                    batch_size = test_batch_size,
                                                    collate_fn = collate_fn_BEV_test,
                                                    shuffle = False,
                                                    num_workers = 4)
    val_dataset_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                                    batch_size = test_batch_size,
                                                    collate_fn = collate_fn_BEV,
                                                    shuffle = False,
                                                    num_workers = 4)

    # validation
    print('*'*80)
    print('Test network performance on validation split')
    print('*'*80)
    pbar = tqdm(total=len(val_dataset_loader))
    time_list = []
    pp_time_list = []
    evaluator = PanopticEval(len(unique_label)+1, None, [0], min_points=50)
    with torch.no_grad():
        for i_iter_val,(val_vox_fea,val_vox_label,val_gt_center,val_gt_offset,val_grid,val_pt_labels,val_pt_ints,val_pt_fea) in enumerate(val_dataset_loader):
            val_vox_fea_ten = val_vox_fea.to(pytorch_device)
            val_vox_label = SemKITTI2train(val_vox_label)
            val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in val_pt_fea]
            val_grid_ten = [torch.from_numpy(i[:,:2]).to(pytorch_device) for i in val_grid]
            val_label_tensor=val_vox_label.type(torch.LongTensor).to(pytorch_device)
            val_gt_center_tensor = val_gt_center.to(pytorch_device)
            val_gt_offset_tensor = val_gt_offset.to(pytorch_device)

            torch.cuda.synchronize()
            start_time = time.time()
            if visibility:            
                predict_labels,center,offset = my_model(val_pt_fea_ten, val_grid_ten, val_vox_fea_ten)
            else:
                predict_labels,center,offset = my_model(val_pt_fea_ten, val_grid_ten)
            torch.cuda.synchronize()
            time_list.append(time.time()-start_time)

            for count,i_val_grid in enumerate(val_grid):
                # get foreground_mask
                for_mask = torch.zeros(1,grid_size[0],grid_size[1],grid_size[2],dtype=torch.bool).to(pytorch_device)
                for_mask[0,val_grid[count][:,0],val_grid[count][:,1],val_grid[count][:,2]] = True
                # post processing
                torch.cuda.synchronize()
                start_time = time.time()
                panoptic_labels,center_points = get_panoptic_segmentation(torch.unsqueeze(predict_labels[count], 0),torch.unsqueeze(center[count], 0),torch.unsqueeze(offset[count], 0),val_pt_dataset.thing_list,\
                                                                          threshold=args['model']['post_proc']['threshold'], nms_kernel=args['model']['post_proc']['nms_kernel'],\
                                                                          top_k=args['model']['post_proc']['top_k'], polar=circular_padding,foreground_mask=for_mask)
                torch.cuda.synchronize()
                pp_time_list.append(time.time()-start_time)
                panoptic_labels = panoptic_labels.cpu().detach().numpy().astype(np.uint32)
                panoptic = panoptic_labels[0,val_grid[count][:,0],val_grid[count][:,1],val_grid[count][:,2]]

                evaluator.addBatch(panoptic & 0xFFFF,panoptic,np.squeeze(val_pt_labels[count]),np.squeeze(val_pt_ints[count]))
            del val_vox_label,val_pt_fea_ten,val_label_tensor,val_grid_ten,val_gt_center,val_gt_center_tensor,val_gt_offset,val_gt_offset_tensor,predict_labels,center,offset,panoptic_labels,center_points
            pbar.update(1)
    
    class_PQ, class_SQ, class_RQ, class_all_PQ, class_all_SQ, class_all_RQ = evaluator.getPQ()
    miou,ious = evaluator.getSemIoU()
    print('Validation per class PQ, SQ, RQ and IoU: ')
    for class_name, class_pq, class_sq, class_rq, class_iou in zip(unique_label_str,class_all_PQ[1:],class_all_SQ[1:],class_all_RQ[1:],ious[1:]):
        print('%15s : %6.2f%%  %6.2f%%  %6.2f%%  %6.2f%%' % (class_name, class_pq*100, class_sq*100, class_rq*100, class_iou*100))
    pbar.close()
    print('Current val PQ is %.3f' %
        (class_PQ*100))               
    print('Current val miou is %.3f'%
        (miou*100))
    print('Inference time per %d is %.4f seconds\n, postprocessing time is %.4f seconds per scan' %
        (test_batch_size,np.mean(time_list),np.mean(pp_time_list)))
    
    # test
    print('*'*80)
    print('Generate predictions for test split')
    print('*'*80)
    pbar = tqdm(total=len(test_dataset_loader))
    with torch.no_grad():
        for i_iter_test,(test_vox_fea,_,_,_,test_grid,_,_,test_pt_fea,test_index) in enumerate(test_dataset_loader):
            # predict
            test_vox_fea_ten = test_vox_fea.to(pytorch_device)
            test_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in test_pt_fea]
            test_grid_ten = [torch.from_numpy(i[:,:2]).to(pytorch_device) for i in test_grid]

            if visibility:
                predict_labels,center,offset = my_model(test_pt_fea_ten,test_grid_ten,test_vox_fea_ten)
            else:
                predict_labels,center,offset = my_model(test_pt_fea_ten,test_grid_ten)
            # write to label file
            for count,i_test_grid in enumerate(test_grid):
                # get foreground_mask
                for_mask = torch.zeros(1,grid_size[0],grid_size[1],grid_size[2],dtype=torch.bool).to(pytorch_device)
                for_mask[0,test_grid[count][:,0],test_grid[count][:,1],test_grid[count][:,2]] = True
                # post processing
                panoptic_labels,center_points = get_panoptic_segmentation(torch.unsqueeze(predict_labels[count], 0),torch.unsqueeze(center[count], 0),torch.unsqueeze(offset[count], 0),test_pt_dataset.thing_list,\
                                                                                          threshold=args['model']['post_proc']['threshold'], nms_kernel=args['model']['post_proc']['nms_kernel'],\
                                                                                          top_k=args['model']['post_proc']['top_k'], polar=circular_padding,foreground_mask=for_mask)
                panoptic_labels = panoptic_labels.cpu().detach().numpy().astype(np.uint32)
                panoptic = panoptic_labels[0,test_grid[count][:,0],test_grid[count][:,1],test_grid[count][:,2]]
                save_dir = test_pt_dataset.im_idx[test_index[count]]
                _,dir2 = save_dir.split('/sequences/',1)
                new_save_dir = output_path + '/sequences/' +dir2.replace('velodyne','predictions')[:-3]+'label'
                if not os.path.exists(os.path.dirname(new_save_dir)):
                    try:
                        os.makedirs(os.path.dirname(new_save_dir))
                    except OSError as exc:
                        if exc.errno != errno.EEXIST:
                            raise
                panoptic.tofile(new_save_dir)
            del test_pt_fea_ten,test_grid_ten,test_pt_fea,predict_labels,center,offset
            pbar.update(1)
    pbar.close()
    print('Predicted test labels are saved in %s. Need to be shifted to original label format before submitting to the Competition website.' % output_path)
    print('Remapping script can be found in semantic-kitti-api.')

if __name__ == '__main__':
    # Testing settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--data_dir', default='data')
    parser.add_argument('-p', '--pretrained_model', default='pretrained_weight/Panoptic_SemKITTI_PolarNet.pt')
    parser.add_argument('-c', '--configs', default='configs/SemanticKITTI_model/Panoptic-PolarNet.yaml')
    
    args = parser.parse_args()
    with open(args.configs, 'r') as s:
        new_args = yaml.safe_load(s)
    args = merge_configs(args,new_args)

    print(' '.join(sys.argv))
    print(args)
    main(args)