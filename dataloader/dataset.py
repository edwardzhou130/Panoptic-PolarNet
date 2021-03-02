#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SemKITTI dataloader
"""
import os
import numpy as np
import torch
import random
import time
import numba as nb
import yaml
import pickle
import errno
from torch.utils import data

from .process_panoptic import PanopticLabelGenerator
from .instance_augmentation import instance_augmentation

class SemKITTI(data.Dataset):
    def __init__(self, data_path, imageset = 'train', return_ref = False, instance_pkl_path ='data'):
        self.return_ref = return_ref
        with open("semantic-kitti.yaml", 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        thing_class = semkittiyaml['thing_class']
        self.thing_list = [cl for cl, ignored in thing_class.items() if ignored]
        self.imageset = imageset
        if imageset == 'train':
            split = semkittiyaml['split']['train']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')
        
        self.im_idx = []
        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path,str(i_folder).zfill(2),'velodyne']))
        self.im_idx.sort()

        # get class distribution weight 
        epsilon_w = 0.001
        origin_class = semkittiyaml['content'].keys()
        weights = np.zeros((len(semkittiyaml['learning_map_inv'])-1,),dtype = np.float32)
        for class_num in origin_class:
            if semkittiyaml['learning_map'][class_num] != 0:
                weights[semkittiyaml['learning_map'][class_num]-1] += semkittiyaml['content'][class_num]
        self.CLS_LOSS_WEIGHT = 1/(weights + epsilon_w)
        self.instance_pkl_path = instance_pkl_path
         
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)
    
    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        if self.imageset == 'test':
            sem_data = np.expand_dims(np.zeros_like(raw_data[:,0],dtype=int),axis=1)
            inst_data = np.expand_dims(np.zeros_like(raw_data[:,0],dtype=np.uint32),axis=1)
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne','labels')[:-3]+'label', dtype=np.uint32).reshape((-1,1))
            sem_data = annotated_data & 0xFFFF #delete high 16 digits binary
            sem_data = np.vectorize(self.learning_map.__getitem__)(sem_data)
            inst_data = annotated_data
        data_tuple = (raw_data[:,:3], sem_data.astype(np.uint8),inst_data)
        if self.return_ref:
            data_tuple += (raw_data[:,3],)
        return data_tuple

    def save_instance(self, out_dir, min_points = 10):
        'instance data preparation'
        instance_dict={label:[] for label in self.thing_list}
        for data_path in self.im_idx:
            print('process instance for:'+data_path)
            # get x,y,z,ref,semantic label and instance label
            raw_data = np.fromfile(data_path, dtype=np.float32).reshape((-1, 4))
            annotated_data = np.fromfile(data_path.replace('velodyne','labels')[:-3]+'label', dtype=np.uint32).reshape((-1,1))
            sem_data = annotated_data & 0xFFFF #delete high 16 digits binary
            sem_data = np.vectorize(self.learning_map.__getitem__)(sem_data)
            inst_data = annotated_data

            # instance mask
            mask = np.zeros_like(sem_data,dtype=bool)
            for label in self.thing_list:
                mask[sem_data == label] = True

            # create unqiue instance list
            inst_label = inst_data[mask].squeeze()
            unique_label = np.unique(inst_label)
            num_inst = len(unique_label)

            inst_count = 0
            for inst in unique_label:
                # get instance index
                index = np.where(inst_data == inst)[0]
                # get semantic label
                class_label = sem_data[index[0]]
                # skip small instance
                if index.size<min_points: continue
                # save
                _,dir2 = data_path.split('/sequences/',1)
                new_save_dir = out_dir + '/sequences/' +dir2.replace('velodyne','instance')[:-4]+'_'+str(inst_count)+'.bin'
                if not os.path.exists(os.path.dirname(new_save_dir)):
                    try:
                        os.makedirs(os.path.dirname(new_save_dir))
                    except OSError as exc:
                        if exc.errno != errno.EEXIST:
                            raise
                inst_fea = raw_data[index]
                inst_fea.tofile(new_save_dir)
                instance_dict[int(class_label)].append(new_save_dir)
                inst_count+=1
        with open(out_dir+'/instance_path.pkl', 'wb') as f:
            pickle.dump(instance_dict, f)

def absoluteFilePaths(directory):
   for dirpath,_,filenames in os.walk(directory):
       for f in filenames:
           yield os.path.abspath(os.path.join(dirpath, f))

class voxel_dataset(data.Dataset):
  def __init__(self, in_dataset, args, grid_size, ignore_label = 0, return_test = False, fixed_volume_space= True, use_aug = False, max_volume_space = [50,50,1.5], min_volume_space = [-50,-50,-3]):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = args['rotate_aug'] if use_aug else False
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.flip_aug = args['flip_aug'] if use_aug else False
        self.instance_aug = args['inst_aug'] if use_aug else False
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

        self.panoptic_proc = PanopticLabelGenerator(self.grid_size,sigma=args['gt_generator']['sigma'])
        if self.instance_aug:
            self.inst_aug = instance_augmentation(self.point_cloud_dataset.instance_pkl_path+'/instance_path.pkl',self.point_cloud_dataset.thing_list,self.point_cloud_dataset.CLS_LOSS_WEIGHT,\
                                                random_flip=args['inst_aug_type']['inst_global_aug'],random_add=args['inst_aug_type']['inst_os'],\
                                                random_rotate=args['inst_aug_type']['inst_global_aug'],local_transformation=args['inst_aug_type']['inst_loc_aug'])

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

  def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        if len(data) == 3:
            xyz,labels,insts = data
        elif len(data) == 4:
            xyz,labels,insts,feat = data
            if len(feat.shape) == 1: feat = feat[..., np.newaxis]
        else: raise Exception('Return invalid data tuple')
        if len(labels.shape) == 1: labels = labels[..., np.newaxis]
        if len(insts.shape) == 1: insts = insts[..., np.newaxis]
        
        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random()*360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:,:2] = np.dot( xyz[:,:2],j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4,1)
            if flip_type==1:
                xyz[:,0] = -xyz[:,0]
            elif flip_type==2:
                xyz[:,1] = -xyz[:,1]
            elif flip_type==3:
                xyz[:,:2] = -xyz[:,:2]

        # random instance augmentation
        if self.instance_aug:
            xyz,labels,insts,feat = self.inst_aug.instance_aug(xyz,labels.squeeze(),insts.squeeze(),feat)

        max_bound = np.percentile(xyz,100,axis = 0)
        min_bound = np.percentile(xyz,0,axis = 0)
        
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)

        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        
        intervals = crop_range/(cur_grid_size-1)
        if (intervals==0).any(): print("Zero interval!")
        
        grid_ind = (np.floor((np.clip(xyz,min_bound,max_bound)-min_bound)/intervals)).astype(np.int)

        # process voxel position
        voxel_position = np.zeros(self.grid_size,dtype = np.float32)
        dim_array = np.ones(len(self.grid_size)+1,int)
        dim_array[0] = -1 
        voxel_position = (np.indices(self.grid_size) + 0.5)*intervals.reshape(dim_array) + min_bound.reshape(dim_array)

        # process labels
        processed_label = np.ones(self.grid_size,dtype = np.uint8)*self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind,labels],axis = 1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:,0],grid_ind[:,1],grid_ind[:,2])),:]
        processed_label = nb_process_label(np.copy(processed_label),label_voxel_pair)

        # get thing points mask
        mask = np.zeros_like(labels,dtype=bool)
        for label in self.point_cloud_dataset.thing_list:
            mask[labels == label] = True
        
        inst_label = insts[mask].squeeze()
        unique_label = np.unique(inst_label)
        unique_label_dict = {label:idx+1 for idx , label in enumerate(unique_label)}
        if inst_label.size > 1:            
            inst_label = np.vectorize(unique_label_dict.__getitem__)(inst_label)
            
            # process panoptic
            processed_inst = np.ones(self.grid_size[:2],dtype = np.uint8)*self.ignore_label
            inst_voxel_pair = np.concatenate([grid_ind[mask[:,0],:2],inst_label[..., np.newaxis]],axis = 1)
            inst_voxel_pair = inst_voxel_pair[np.lexsort((grid_ind[mask[:,0],0],grid_ind[mask[:,0],1])),:]
            processed_inst = nb_process_inst(np.copy(processed_inst),inst_voxel_pair)
        else:
            processed_inst = None

        center,center_points,offset = self.panoptic_proc(insts[mask],xyz[mask[:,0]],processed_inst,voxel_position[:2,:,:,0],unique_label_dict,min_bound,intervals)
        
        data_tuple = (voxel_position,processed_label,center,offset)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5)*intervals + min_bound
        return_xyz = xyz - voxel_centers
        return_xyz = np.concatenate((return_xyz,xyz),axis = 1)
        
        if len(data) == 3:
            return_fea = return_xyz
        elif len(data) == 4:
            return_fea = np.concatenate((return_xyz,feat),axis = 1)
        
        if self.return_test:
            data_tuple += (grid_ind,labels,insts,return_fea,index)
        else:
            data_tuple += (grid_ind,labels,insts,return_fea)
        return data_tuple

# transformation between Cartesian coordinates and polar coordinates
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:,0]**2 + input_xyz[:,1]**2)
    phi = np.arctan2(input_xyz[:,1],input_xyz[:,0])
    return np.stack((rho,phi,input_xyz[:,2]),axis=1)

def polar2cat(input_xyz_polar):
    x = input_xyz_polar[0]*np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0]*np.sin(input_xyz_polar[1])
    return np.stack((x,y,input_xyz_polar[2]),axis=0)

class spherical_dataset(data.Dataset):
  def __init__(self, in_dataset, args, grid_size, ignore_label = 0, return_test = False, use_aug = False, fixed_volume_space= True, max_volume_space = [50,np.pi,1.5], min_volume_space = [3,-np.pi,-3]):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = args['rotate_aug'] if use_aug else False
        self.flip_aug = args['flip_aug'] if use_aug else False
        self.instance_aug = args['inst_aug'] if use_aug else False
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

        self.panoptic_proc = PanopticLabelGenerator(self.grid_size,sigma=args['gt_generator']['sigma'],polar=True)
        if self.instance_aug:
            self.inst_aug = instance_augmentation(self.point_cloud_dataset.instance_pkl_path+'/instance_path.pkl',self.point_cloud_dataset.thing_list,self.point_cloud_dataset.CLS_LOSS_WEIGHT,\
                                                random_flip=args['inst_aug_type']['inst_global_aug'],random_add=args['inst_aug_type']['inst_os'],\
                                                random_rotate=args['inst_aug_type']['inst_global_aug'],local_transformation=args['inst_aug_type']['inst_loc_aug'])

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

  def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        if len(data) == 3:
            xyz,labels,insts = data
        elif len(data) == 4:
            xyz,labels,insts,feat = data
            if len(feat.shape) == 1: feat = feat[..., np.newaxis]
        else: raise Exception('Return invalid data tuple')
        if len(labels.shape) == 1: labels = labels[..., np.newaxis]
        if len(insts.shape) == 1: insts = insts[..., np.newaxis]
        
        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random()*360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:,:2] = np.dot( xyz[:,:2],j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4,1)
            if flip_type==1:
                xyz[:,0] = -xyz[:,0]
            elif flip_type==2:
                xyz[:,1] = -xyz[:,1]
            elif flip_type==3:
                xyz[:,:2] = -xyz[:,:2]

        # random instance augmentation
        if self.instance_aug:
            xyz,labels,insts,feat = self.inst_aug.instance_aug(xyz,labels.squeeze(),insts.squeeze(),feat)
        
        # convert coordinate into polar coordinates
        xyz_pol = cart2polar(xyz)
        
        max_bound_r = np.percentile(xyz_pol[:,0],100,axis = 0)
        min_bound_r = np.percentile(xyz_pol[:,0],0,axis = 0)
        max_bound = np.max(xyz_pol[:,1:],axis = 0)
        min_bound = np.min(xyz_pol[:,1:],axis = 0)
        max_bound = np.concatenate(([max_bound_r],max_bound))
        min_bound = np.concatenate(([min_bound_r],min_bound))
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)

        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range/(cur_grid_size-1)

        if (intervals==0).any(): print("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol,min_bound,max_bound)-min_bound)/intervals)).astype(np.int)

        current_grid = grid_ind[:np.size(labels)]

        # process voxel position
        voxel_position = np.zeros(self.grid_size,dtype = np.float32)
        dim_array = np.ones(len(self.grid_size)+1,int)
        dim_array[0] = -1 
        voxel_position = np.indices(self.grid_size)*intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        # voxel_position = polar2cat(voxel_position)
        
        # process labels
        processed_label = np.ones(self.grid_size,dtype = np.uint8)*self.ignore_label
        label_voxel_pair = np.concatenate([current_grid,labels],axis = 1)
        label_voxel_pair = label_voxel_pair[np.lexsort((current_grid[:,0],current_grid[:,1],current_grid[:,2])),:]
        processed_label = nb_process_label(np.copy(processed_label),label_voxel_pair)
        # data_tuple = (voxel_position,processed_label)

        # get thing points mask
        mask = np.zeros_like(labels,dtype=bool)
        for label in self.point_cloud_dataset.thing_list:
            mask[labels == label] = True
        
        inst_label = insts[mask].squeeze()
        unique_label = np.unique(inst_label)
        unique_label_dict = {label:idx+1 for idx , label in enumerate(unique_label)}
        if inst_label.size > 1:            
            inst_label = np.vectorize(unique_label_dict.__getitem__)(inst_label)
            
            # process panoptic
            processed_inst = np.ones(self.grid_size[:2],dtype = np.uint8)*self.ignore_label
            inst_voxel_pair = np.concatenate([current_grid[mask[:,0],:2],inst_label[..., np.newaxis]],axis = 1)
            inst_voxel_pair = inst_voxel_pair[np.lexsort((current_grid[mask[:,0],0],current_grid[mask[:,0],1])),:]
            processed_inst = nb_process_inst(np.copy(processed_inst),inst_voxel_pair)
        else:
            processed_inst = None

        center,center_points,offset = self.panoptic_proc(insts[mask],xyz[:np.size(labels)][mask[:,0]],processed_inst,voxel_position[:2,:,:,0],unique_label_dict,min_bound,intervals)

        # prepare visiblity feature
        # find max distance index in each angle,height pair
        valid_label = np.zeros_like(processed_label,dtype=bool)
        valid_label[current_grid[:,0],current_grid[:,1],current_grid[:,2]] = True
        valid_label = valid_label[::-1]
        max_distance_index = np.argmax(valid_label,axis=0)
        max_distance = max_bound[0]-intervals[0]*(max_distance_index)
        distance_feature = np.expand_dims(max_distance, axis=2)-np.transpose(voxel_position[0],(1,2,0))
        distance_feature = np.transpose(distance_feature,(1,2,0))
        # convert to boolean feature
        distance_feature = (distance_feature>0)*-1.
        distance_feature[current_grid[:,2],current_grid[:,0],current_grid[:,1]]=1.

        data_tuple = (distance_feature,processed_label,center,offset)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5)*intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz,xyz_pol,xyz[:,:2]),axis = 1)

        if len(data) == 3:
            return_fea = return_xyz
        elif len(data) == 4:
            return_fea = np.concatenate((return_xyz,feat),axis = 1)
        
        if self.return_test:
            data_tuple += (grid_ind,labels,insts,return_fea,index)
        else:
            data_tuple += (grid_ind,labels,insts,return_fea)
        return data_tuple
    
@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])',nopython=True,cache=True,parallel = False)
def nb_process_label(processed_label,sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,),dtype = np.uint16)
    counter[sorted_label_voxel_pair[0,3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0,:3]
    for i in range(1,sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i,:3]
        if not np.all(np.equal(cur_ind,cur_sear_ind)):
            processed_label[cur_sear_ind[0],cur_sear_ind[1],cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,),dtype = np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i,3]] += 1
    processed_label[cur_sear_ind[0],cur_sear_ind[1],cur_sear_ind[2]] = np.argmax(counter)
    return processed_label

@nb.jit('u1[:,:](u1[:,:],i8[:,:])',nopython=True,cache=True,parallel = False)
def nb_process_inst(processed_inst,sorted_inst_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,),dtype = np.uint16)
    counter[sorted_inst_voxel_pair[0,2]] = 1
    cur_sear_ind = sorted_inst_voxel_pair[0,:2]
    for i in range(1,sorted_inst_voxel_pair.shape[0]):
        cur_ind = sorted_inst_voxel_pair[i,:2]
        if not np.all(np.equal(cur_ind,cur_sear_ind)):
            processed_inst[cur_sear_ind[0],cur_sear_ind[1]] = np.argmax(counter)
            counter = np.zeros((label_size,),dtype = np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_inst_voxel_pair[i,2]] += 1
    processed_inst[cur_sear_ind[0],cur_sear_ind[1]] = np.argmax(counter)
    return processed_inst

def collate_fn_BEV(data):
    data2stack=np.stack([d[0] for d in data]).astype(np.float32)
    label2stack=np.stack([d[1] for d in data])
    center2stack=np.stack([d[2] for d in data])
    offset2stack=np.stack([d[3] for d in data])
    grid_ind_stack = [d[4] for d in data]
    point_label = [d[5] for d in data]
    point_inst = [d[6] for d in data]
    xyz = [d[7] for d in data]
    return torch.from_numpy(data2stack),torch.from_numpy(label2stack),torch.from_numpy(center2stack),torch.from_numpy(offset2stack),grid_ind_stack,point_label,point_inst,xyz

def collate_fn_BEV_test(data):    
    data2stack=np.stack([d[0] for d in data]).astype(np.float32)
    label2stack=np.stack([d[1] for d in data])
    center2stack=np.stack([d[2] for d in data])
    offset2stack=np.stack([d[3] for d in data])
    grid_ind_stack = [d[4] for d in data]
    point_label = [d[5] for d in data]
    point_inst = [d[6] for d in data]
    xyz = [d[7] for d in data]
    index = [d[8] for d in data]
    return torch.from_numpy(data2stack),torch.from_numpy(label2stack),torch.from_numpy(center2stack),torch.from_numpy(offset2stack),grid_ind_stack,point_label,point_inst,xyz,index

# load Semantic KITTI class info
with open("semantic-kitti.yaml", 'r') as stream:
    semkittiyaml = yaml.safe_load(stream)
SemKITTI_label_name = dict()
for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
    SemKITTI_label_name[semkittiyaml['learning_map'][i]] = semkittiyaml['labels'][i]