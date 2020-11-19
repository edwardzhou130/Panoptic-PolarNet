#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

class PanopticLabelGenerator(object):
    def __init__(self,grid_size,sigma=5,polar=False):
        """Initialize panoptic ground truth generator

        Args:
            grid_size: voxel size.
            sigma (int, optional):  Gaussian distribution paramter. Create heatmap in +-3*sigma window. Defaults to 5.
            polar (bool, optional): Is under polar coordinate. Defaults to False.
        """        
        self.grid_size = grid_size
        self.polar = polar

        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    
    def __call__(self,inst,xyz,voxel_inst,voxel_position,label_dict,min_bound,intervals):
        """Generate instance center and offset ground truth

        Args:
            inst : instance panoptic label (N)
            xyz : point location (N x 3)
            voxel_inst : voxel panoptic label on the BEV (H x W)
            voxel_position : voxel location on the BEV (3 x H x W)
            label_dict : unqiue instance label dict
            min_bound : space minimal bound
            intervals : voxelization intervals

        Returns:
            center, center_pts, offset
        """        
        height, width = self.grid_size[0],self.grid_size[1]

        center = np.zeros((1, height, width), dtype=np.float32)
        center_pts = []
        offset = np.zeros((2, height, width), dtype=np.float32)
        #skip empty instances
        if inst.size < 2: return center, center_pts, offset
        # find unique instances
        inst_labels = np.unique(inst)
        for inst_label in inst_labels:
            # get mask for each unique instance
            mask = np.where(inst == inst_label)
            voxel_mask = np.where(voxel_inst == label_dict[inst_label])
            # get center
            center_x, center_y = np.mean(xyz[mask,0]), np.mean(xyz[mask,1])
            if self.polar:
                # convert to polar coordinate
                center_x_pol, center_y_pol = np.sqrt(center_x**2 + center_y**2),np.arctan2(center_y,center_x)
                center_x = center_x_pol
                center_y = center_y_pol

            # generate center heatmap
            x, y = int(np.floor((center_x-min_bound[0])/intervals[0])), int(np.floor((center_y-min_bound[1])/intervals[1]))
            center_pts.append([x, y])
            # outside image boundary
            if x < 0 or y < 0 or \
                    x >= height or y >= width:
                continue
            sigma = self.sigma
            # upper left
            ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
            # bottom right
            br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

            if self.polar:
                c, d = max(0, -ul[0]), min(br[0], height) - ul[0]
                a, b = 0, br[1] - ul[1]

                cc, dd = max(0, ul[0]), min(br[0], height)
                angle_list = [angle_id % width for angle_id in range(ul[1],br[1])]
                center[0, cc:dd, angle_list] = np.maximum(
                    center[0, cc:dd, angle_list], np.transpose(self.g[c:d,a:b]))
            else:
                c, d = max(0, -ul[0]), min(br[0], height) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], width) - ul[1]

                cc, dd = max(0, ul[0]), min(br[0], height)
                aa, bb = max(0, ul[1]), min(br[1], width)
                center[0, cc:dd, aa:bb] = np.maximum(
                    center[0, cc:dd, aa:bb], self.g[c:d,a:b])

            if self.polar:
                # generate offset (2, h, w) -> (y-dir, x-dir)
                offset[0,voxel_mask[0],voxel_mask[1]] = (center_x - voxel_position[0,voxel_mask[0],voxel_mask[1]])/intervals[0]
                offset[1,voxel_mask[0],voxel_mask[1]] = ((center_y - voxel_position[1,voxel_mask[0],voxel_mask[1]]+np.pi)%(2*np.pi) - np.pi)/intervals[1]
            else:
                # generate offset (2, h, w) -> (y-dir, x-dir)
                offset[0,voxel_mask[0],voxel_mask[1]] = (center_x - voxel_position[0,voxel_mask[0],voxel_mask[1]])/intervals[0]
                offset[1,voxel_mask[0],voxel_mask[1]] = (center_y - voxel_position[1,voxel_mask[0],voxel_mask[1]])/intervals[1]

        # print('gt center',center_pts)

        return center, center_pts, offset