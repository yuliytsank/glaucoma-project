#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import numpy as np
from utils import visualize_slice_CAM_glaucoma3d
import pandas as pd

load_dir = os.path.expanduser(os.path.join('..', 'Images', 'Data', 'NP_Volumes', 'subsampled_all'))
save_dir = os.path.expanduser(os.path.join('..', 'Images', 'Data', 'NP_Volumes', 'precomputed_CAMs'))
num_runs = 5

stats_all = np.load(os.path.join('..','custom_3d_stats_all_glaucoma.npy')).item()
train_setSize = 4

current_mean = np.mean(stats_all['perform']['val'][:,:,train_setSize][:,:-1], 1)
current_std = np.std(stats_all['perform']['val'][:,:,train_setSize][:,:-1], 1)

max_ind = np.argmax(current_mean)
end_ind = int(np.argwhere(np.isnan(stats_all['preds']['test'][max_ind, 0, train_setSize,:]))[0])-1

precomputed_CAM_info = {}

precomputed_CAM_info['all_probs'] = []
precomputed_CAM_info['all_targets'] = []
precomputed_CAM_info['all_paths'] = []

if not os.path.exists(save_dir):
        os.mkdir(save_dir)

for cv_fold in range(0,num_runs):
    
    paths = stats_all['paths']['test'][max_ind, cv_fold, train_setSize, 0:end_ind]
    targets = stats_all['targets']['test'][max_ind, cv_fold, train_setSize, 0:end_ind]
    probs = stats_all['probs']['test'][max_ind, cv_fold, train_setSize, 0:end_ind]
    
    precomputed_CAM_info['all_paths'].extend(paths.tolist())
    precomputed_CAM_info['all_targets'].extend(targets.tolist())
    precomputed_CAM_info['all_probs'].extend(probs.tolist())
    
    precomputed_CAM_df = pd.DataFrame(precomputed_CAM_info)

precomputed_CAM_df.to_pickle('precomputed_CAM_df')    

for sample_num in range(0, len(precomputed_CAM_info['all_targets'])):
    
    load_path = os.path.join(load_dir, os.path.split(precomputed_CAM_info['all_paths'][sample_num])[-1])
    np_contents = np.load(load_path)
    CAM_slices_overlaid, original_slices, probs = visualize_slice_CAM_glaucoma3d.getCAM(np_contents, sample_num)
    
    save_path = os.path.join(save_dir, os.path.split(load_path)[-1])
    
    precomputed_CAM_info['all_paths'][sample_num] = save_path
    np.save(save_path, CAM_slices_overlaid)
    
np.save('precomputed_CAM_info', precomputed_CAM_info)