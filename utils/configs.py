#!/usr/bin/env python3
import yaml

def merge_configs(cfgs,new_cfgs):
    if hasattr(cfgs, 'data_dir'):
        new_cfgs['dataset']['path']=cfgs.data_dir
    if hasattr(cfgs, 'model_save_path'):
        new_cfgs['model']['model_save_path']=cfgs.model_save_path
    if hasattr(cfgs, 'pretrained_model'):
        new_cfgs['model']['pretrained_model']=cfgs.pretrained_model
    return new_cfgs