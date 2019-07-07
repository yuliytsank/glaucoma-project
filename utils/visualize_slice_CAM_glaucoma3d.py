#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from torch.autograd import Variable
import numpy as np

selected_layers = [3]
gap_layer = 5
softmax = nn.Softmax()
batch_size = 1

use_cuda = 0
if torch.cuda.is_available():
    use_cuda =1

num_classes = 2

import custom3d_GAP_glaucoma
model = custom3d_GAP_glaucoma.Glaucoma3d(num_classes = num_classes)

map_width = 21
map_depth = 7

original_dims = (22,64,64)
new_dims = (map_depth, map_width, map_width)

run_num = 0

max_ind_epochs = np.array([46])
max_ind_epochs = max_ind_epochs.astype(np.uint8)

trainSet_sizes = ['all']

def getCAM(np_contents, sample_num=0):
       
    for trainSet_size_num, trainSet_size in enumerate(trainSet_sizes):
    
        for test_set_num in range(0,1):
            
            correct_ans_avg = {}
            
            for run_num in range(0,1):
                
                correct_ans_avg['run'+str(run_num)] = {}
                model = custom3d_GAP_glaucoma.Glaucoma3d(num_classes = 2)
                
                if use_cuda:
                    model.cuda()
                    device = torch.device('cuda')
                else:
                    device = torch.device('cpu') 
            
                path_saved_model =os.path.join('.', 'model_glaucoma_'+ trainSet_size +'_epoch_'+str(max_ind_epochs[trainSet_size_num, test_set_num])+'_run_' +str(run_num))
                  
                state_dict = torch.load(path_saved_model,map_location=device)

                model.load_state_dict(state_dict)
             
                model.features = nn.Sequential(*list(model.children())[:-2])
                model.eval()
         
                test_batch = 0

                num_test_samples = 1

                weighted_classes_summed = {}
                weighted_classes = {}
                
                for class_num in range(0, num_classes):
                    
                    weighted_classes_summed['class'+str(class_num+1)] = np.empty((map_depth,map_width,map_width,num_test_samples))*np.nan
                
                targets = np.empty(num_test_samples)*np.nan
                preds = np.empty(num_test_samples)*np.nan
                probs = np.empty(num_test_samples)*np.nan
 
                target_idx = 0
                target = 1#find probability of glaucoma
                np_data = np_contents
                data = torch.from_numpy(np_data[None,None,:,:,:]).float()
                print('SampleNum: '+str(sample_num) +'/'+str(num_test_samples))
  
                targets[target_idx] = target
                
                test_batch+=1
                if use_cuda:
                    data = data.cuda()
    #            data, target = Variable(data, volatile=True), Variable(target)
                data = Variable(data, volatile=True)
    
                output = model(data)
                preds[target_idx] = output.data.max(1)[1].clone().cpu().detach().numpy()
                probs[target_idx] = softmax(output[0])[target]
                probs = np.around(probs,2)

                params = list(model.parameters())[16]
                weights_np = params.clone().cpu().detach().numpy()

                x = data[:,:,:,:,:]
            #    chan_depth = x.shape[1]
                for index, layer in enumerate(model.features):
                    # Forward pass layer by layer
                    x = layer(x)
            #        print('Layer '+str(index))
    #                    selected_layer = index
                    if index in selected_layers:

                        x_np = x.clone().cpu().detach().numpy()#convert torch tensor to numpy

                CAM_slices_overlaid = {}
                original_slices = {}
                
                weighted_classes['class'+str(target)] = x_np[0,:,:,:,:]*weights_np[target,:,None,None,None]
                weighted_classes_summed['class'+str(target)][:,:,:,target_idx] = np.sum(weighted_classes['class'+str(target)],0)
                
    #            interpolated_CAM = resize(weighted_classes_summed['class'+str(target)][:,:,:,target_idx], original_dims)
                interpolated_CAM = zoom(weighted_classes_summed['class'+str(target)][:,:,:,target_idx], 3.0)
                
                sampled_slices_CAM = {}
                
                sampled_slices_CAM['enface'] = interpolated_CAM[[0,10,20],:,:]
                sampled_slices_CAM['cross_section'] = interpolated_CAM[:,(5,30,55),0:63]
    #                        img = [[0.9, 0.3], [0.2, 0.1]]
                cmap = plt.get_cmap('jet')
                
                original_slices['enface'] = np_data[(0,10,20),0:63,0:63]
                original_slices['cross_section'] = np_data[0:21,(5,30,55),0:63]

#                transp_val = 0.15
                slice_views = ['enface', 'cross_section']
                for slice_view in slice_views:
                    
                    CAM_slices_overlaid[slice_view]={}
                    for slice_num in range(0,3):
                        
                        if slice_view == 'enface':
                            current_slice = original_slices[slice_view][slice_num,:,:].astype(np.float)/(255.*.7)
                            current_slice_CAM = sampled_slices_CAM['enface'][slice_num,:,:]
                        else:
                            current_slice = original_slices[slice_view][:,slice_num,:].astype(np.float)/(255.*.7)
                            current_slice_CAM = sampled_slices_CAM['cross_section'][:,slice_num,:]
                        
                        current_slice_CAM_rgba = cmap(current_slice_CAM)
                        current_slice_rgb = np.repeat(current_slice[:,:,None], 3,axis=2)
                        
                        current_slice_CAM_rgb = np.delete(current_slice_CAM_rgba, 3, 2).astype(np.float)
                        transp_val = current_slice_CAM/np.max(current_slice_CAM)
                        transp_val = np.repeat(transp_val[:,:,None],3, axis = 2)
                        current_slice_CAM_overlaid = (current_slice_CAM_rgb*transp_val+current_slice_rgb*(1.-transp_val))*255.
                       
                        CAM_slices_overlaid[slice_view]['slice'+str(slice_num)]= current_slice_CAM_overlaid.astype(np.uint8)
                        
    return CAM_slices_overlaid, original_slices, probs
      
                    
  