#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pickle

class instance_augmentation(object):
    def __init__(self,instance_pkl_path,thing_list,class_weight,random_flip = False,random_add = False,random_rotate = False,local_transformation = False):
        self.thing_list = thing_list
        self.instance_weight = [class_weight[thing_class_num-1] for thing_class_num in thing_list]
        self.instance_weight = np.asarray(self.instance_weight)/np.sum(self.instance_weight)
        self.random_flip = random_flip
        self.random_add = random_add
        self.random_rotate = random_rotate
        self.local_transformation = local_transformation

        self.add_num = 5

        with open(instance_pkl_path, 'rb') as f:
            self.instance_path = pickle.load(f)

    def instance_aug(self, point_xyz, point_label, point_inst, point_feat = None):
        """random rotate and flip each instance independently.

        Args:
            point_xyz: [N, 3], point location
            point_label: [N, 1], class label
            point_inst: [N, 1], instance label
        """        
        # random add instance to this scan
        if self.random_add:
            # choose which instance to add
            instance_choice = np.random.choice(len(self.thing_list),self.add_num,replace=True,p=self.instance_weight)
            uni_inst, uni_inst_count = np.unique(instance_choice,return_counts=True)
            add_idx = 1
            total_point_num = 0
            early_break = False
            for n, count in zip(uni_inst, uni_inst_count):
                # find random instance
                random_choice = np.random.choice(len(self.instance_path[self.thing_list[n]]),count)
                # add to current scan
                for idx in random_choice:
                    points = np.fromfile(self.instance_path[self.thing_list[n]][idx], dtype=np.float32).reshape((-1, 4))
                    add_xyz = points[:,:3]
                    center = np.mean(add_xyz,axis=0)

                    # need to check occlusion
                    fail_flag = True
                    if self.random_rotate:
                        # random rotate
                        random_choice = np.random.random(20)*np.pi*2
                        for r in random_choice:
                            center_r = self.rotate_origin(center[np.newaxis,...],r)
                            # check if occluded
                            if self.check_occlusion(point_xyz,center_r[0]):
                                fail_flag = False
                                break
                        # rotate to empty space
                        if fail_flag: continue
                        add_xyz = self.rotate_origin(add_xyz,r)
                    else:
                        fail_flag = not self.check_occlusion(point_xyz,center)
                    if fail_flag: continue

                    add_label = np.ones((points.shape[0],),dtype=np.uint8)*(self.thing_list[n])
                    add_inst = np.ones((points.shape[0],),dtype=np.uint32)*(add_idx<<16)
                    point_xyz = np.concatenate((point_xyz,add_xyz),axis=0)
                    point_label = np.concatenate((point_label,add_label),axis=0)
                    point_inst = np.concatenate((point_inst,add_inst),axis=0)
                    if point_feat is not None:
                        add_fea =  points[:,3:]
                        if len(add_fea.shape) == 1: add_fea = add_fea[..., np.newaxis]
                        point_feat = np.concatenate((point_feat,add_fea),axis=0)
                    add_idx +=1
                    total_point_num += points.shape[0]
                    if total_point_num>5000:
                        early_break=True
                        break
                # prevent adding too many points which cause GPU memory error
                if early_break: break

        # instance mask
        mask = np.zeros_like(point_label,dtype=bool)
        for label in self.thing_list:
            mask[point_label == label] = True

        # create unqiue instance list
        inst_label = point_inst[mask].squeeze()
        unique_label = np.unique(inst_label)
        num_inst = len(unique_label)

        
        for inst in unique_label:
            # get instance index
            index = np.where(point_inst == inst)[0]
            # skip small instance
            if index.size<10: continue
            # get center
            center = np.mean(point_xyz[index,:],axis=0)

            if self.local_transformation:
                # random translation and rotation
                point_xyz[index,:] = self.local_tranform(point_xyz[index,:],center)
            
            # random flip instance based on it center 
            if self.random_flip:
                # get axis
                long_axis = [center[0], center[1]]/(center[0]**2+center[1]**2)**0.5
                short_axis = [-long_axis[1],long_axis[0]]
                # random flip
                flip_type = np.random.choice(5,1)
                if flip_type==3:
                    point_xyz[index,:2] = self.instance_flip(point_xyz[index,:2],[long_axis,short_axis],[center[0], center[1]],flip_type)
            
            # 20% random rotate
            random_num = np.random.random_sample()
            if self.random_rotate:
                if random_num>0.8 and inst & 0xFFFF > 0:
                    random_choice = np.random.random(20)*np.pi*2
                    fail_flag = True
                    for r in random_choice:
                        center_r = self.rotate_origin(center[np.newaxis,...],r)
                        # check if occluded
                        if self.check_occlusion(np.delete(point_xyz, index, axis=0),center_r[0]):
                            fail_flag = False
                            break
                    if not fail_flag:
                        # rotate to empty space
                        point_xyz[index,:] = self.rotate_origin(point_xyz[index,:],r)

        if len(point_label.shape) == 1: point_label = point_label[..., np.newaxis]
        if len(point_inst.shape) == 1: point_inst = point_inst[..., np.newaxis]
        if point_feat is not None:
            return point_xyz,point_label,point_inst,point_feat
        else:
            return point_xyz,point_label,point_inst

    def instance_flip(self, points,axis,center,flip_type = 1):
        points = points[:]-center
        if flip_type == 1:
            # rotate 180 degree
            points = -points+center
        elif flip_type == 2:
            # flip over long axis
            a = axis[0][0]
            b = axis[0][1]
            flip_matrix = np.array([[b**2 - a**2, -2 * a * b],[-2 * a * b, a**2 - b**2]])
            points = np.matmul(flip_matrix,np.transpose(points, (1, 0)))
            points = np.transpose(points, (1, 0))+center
        elif flip_type == 3:
            # flip over short axis
            a = axis[1][0]
            b = axis[1][1]
            flip_matrix = np.array([[b**2 - a**2, -2 * a * b],[-2 * a * b, a**2 - b**2]])
            points = np.matmul(flip_matrix,np.transpose(points, (1, 0)))
            points = np.transpose(points, (1, 0))+center

        return points

    def check_occlusion(self,points,center,min_dist=2):
        'check if close to a point'
        dist = np.linalg.norm(points-center,axis=0)
        return np.all(dist>min_dist)

    def rotate_origin(self,xyz,radians):
        'rotate a point around the origin'
        x = xyz[:,0]
        y = xyz[:,1]
        new_xyz = xyz.copy()
        new_xyz[:,0] = x * np.cos(radians) + y * np.sin(radians)
        new_xyz[:,1] = -x * np.sin(radians) + y * np.cos(radians)
        return new_xyz

    def local_tranform(self,xyz,center):
        'translate and rotate point cloud according to its center'
        # random xyz
        loc_noise = np.random.normal(scale = 0.25, size=(1,3))
        # random angle
        rot_noise = np.random.uniform(-np.pi/20, np.pi/20)

        xyz = xyz-center
        xyz = self.rotate_origin(xyz,rot_noise)
        xyz = xyz+loc_noise
        
        return xyz+center
