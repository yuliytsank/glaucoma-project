#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 12:07:36 2019

@author: yuliy
"""

import os
import numpy as np

extract_OCT_images_path = os.path.expanduser(os.path.join('~', 'Downloads', 'GlaucomaOCT'))

classes = ['Normal', 'POAG']

total_Normal = 263
total_POAG = 847

oversample_normal_times = int(round(total_POAG/total_Normal))

save_dir = os.path.expanduser(os.path.join('~', 'Desktop', 'InsightProjectData', 'NP_Volumes', 'subsampled_all'))
Normal_count = 0
POAG_count = 0 

def get_data_info(path_data_root = extract_OCT_images_path, subsample_vols = 0, save_dir = save_dir):
    
    dir_info = os.listdir(path_data_root)

    data_info = {}
    data_info['paths'] = []
    data_info['targets'] = []
    data_info['classes'] = classes
    data_info['class_to_idx'] = {classes[i]: i for i in range(len(classes))}
    data_info['samples'] = []
    
    if subsample_vols:
        if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
    
    for image_count, image_path in enumerate(dir_info):
    #    image_path = dir_info[0]
        image_name = image_path.split('.')[0]
        image_info_split = image_path.split('-')
        
        full_image_path = os.path.join(path_data_root, image_path)
                
        class_name = image_info_split[0]
        
        class_num = data_info['class_to_idx'][class_name]
        path_to_saved_image = os.path.expanduser(os.path.join('~', 'Desktop', 'InsightProjectData', 'NP_Volumes',save_dir, image_path))
        
        data_info['paths'].append(path_to_saved_image)
        data_info['targets'].append(class_num)
        
        if subsample_vols:
            vol_data = np.load(full_image_path)
            vol_data = vol_data.transpose((1,0,2))
            subsampled_vol_data = vol_data[range(0,128,6),:,:]
            np.save(path_to_saved_image + '.npy', subsampled_vol_data)
        
    return data_info
    
if __name__ == '__main__':
    data_info = get_data_info()
    
    
    
    