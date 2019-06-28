#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 18:20:44 2018

@author: yuliy
"""
#def initialize_everything():
selected_layers = [3]
gap_layer = 5
selected_filter = 5
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
#from skimage.transform import resize
from scipy.ndimage import zoom
#import matplotlib.image as mpimg
#import torch.nn.functional as F
#import torch.optim as optim
#from torchvision import datasets, transforms, models
from torch.autograd import Variable
#import random
import numpy as np
#import cv2
#import scipy.io as sio
#import custom_transforms
#from run_trainTest_3d import OCT_Folder
softmax = nn.Softmax()
batch_size = 1

use_cuda = 0
if torch.cuda.is_available():
    use_cuda =1

#import VGG_code_original    
#model = VGG_code_original.vgg16_bn(num_classes = 2)
num_classes = 2

import custom3d_GAP_glaucoma
model = custom3d_GAP_glaucoma.Glaucoma3d(num_classes = 2)

#if use_cuda:
#    model.cuda()
#    model.features = torch.nn.DataParallel(model.features)
    
map_width = 21
map_depth = 7

original_dims = (22,64,64)
new_dims = (map_depth, map_width, map_width)

run_num = 0

max_ind_epochs = np.array([[46., 0],
       [0, 0],
       [0, 0]])
max_ind_epochs = max_ind_epochs.astype(np.uint8)

#max_ind_epochs = np.array([[0,0, 0, 0], [0, 0, 0, 0],[0, 0, 0, 0]])

inverted_prop_num = 0

#weighted_classes = {}
trainSet_sizes = ['all']




def getCAM(np_contents, sample_num=0):
    
        
#    try:
#        test_sample_path = filename
#    except NameError:
#        test_sample_path = 'POAG-000008-2009-02-03-OD.npy'

    
    for trainSet_size_num, trainSet_size in enumerate(trainSet_sizes):
        
    #    weighted_classes['train_fix'+str(train_fix_num)] = {}
    
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
                
            #############################################################change directory after new run######################
    #        path_saved_model = os.path.join('.', 'models_Mparams', 'model_epoch_'+str(max_ind_epochs[conf_num,fix_pos])+'_'+'fix_'+ str(fix_positions[fix_pos]) + '_' + \
    #                                                                'config' + str(conf) + '_' + 'inv_num' + str(inverted_prop_num)+ '_run' + str(run_num))
            
                path_saved_model =os.path.join('.', 'model_glaucoma_'+ trainSet_size +'_epoch_'+str(max_ind_epochs[trainSet_size_num, test_set_num])+'_run_' +str(run_num))
                
                
                
                state_dict = torch.load(path_saved_model,map_location=device)
                #from collections import OrderedDict
                #new_state_dict = OrderedDict()
                #for k, v in state_dict.items():
                #    name = k[7:] # remove `module.`
                #    new_state_dict[name] = v
                ## load params
                model.load_state_dict(state_dict)
                
                
                model.features = nn.Sequential(*list(model.children())[:-2])
                model.eval()
            #    path_to_test_data = os.path.expanduser(os.path.join('~', 'Desktop', 'HumanaeVGG_train_test_224/set2/fix_'+str(fix_positions[fix_pos])+'/test/'))
    #            path_to_test_data = os.path.expanduser(os.path.join('~', 'Desktop', 'InsightProjectData', 'NP_Volumes','validation'))
    #            
    #            test_loader = torch.utils.data.DataLoader(
    #                OCT_Folder(path_to_test_data),
    #                batch_size=batch_size, shuffle=True)
    #            
                
                
                
                test_batch = 0
                
    #            num_test_samples = len(test_loader)
                num_test_samples = 1
    
                
                weighted_classes_summed = {}
                weighted_classes = {}
                
                for class_num in range(0, num_classes):
                    
                    weighted_classes_summed['class'+str(class_num+1)] = np.empty((map_depth,map_width,map_width,num_test_samples))*np.nan
        #            weighted_class1_all = np.empty((map_width,map_width,num_test_samples))*np.nan
        #            weighted_class2_all = np.empty((map_width,map_width,num_test_samples))*np.nan
        #            weighted_class3_all = np.empty((map_width,map_width,num_test_samples))*np.nan
                
                targets = np.empty(num_test_samples)*np.nan
                preds = np.empty(num_test_samples)*np.nan
                probs = np.empty(num_test_samples)*np.nan
                
                
    #            for target_idx, (data, target) in enumerate(test_loader):
                    
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
            
    #        
    #            for index, params in enumerate(model.parameters()):
    ##                print('Classifier Layer '+str(index))
    #                if index == gap_layer:
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
                        # Only need to forward until the selected layer is reached
                        # Now, x is the output of the selected layer
                      
                        # Here, we get the specific filter from the output of the convolution operation
                        # x is a tensor of shape 1x512x28x28.(For layer 17)
                        # So there are 512 unique filter outputs
                        # Following line selects a filter from 512 filters so self.conv_output will become
    #                    # a tensor of shape 28x28
    #                    chan_depth = x.shape[1]
    #                    chan_height = x.shape[2]
    #                    chan_width = x.shape[3]
                        x_np = x.clone().cpu().detach().numpy()#convert torch tensor to numpy
    #                    output_im_width = np.ceil(np.sqrt(chan_depth))*chan_width
    #                    output_im_height =np.ceil(np.sqrt(chan_depth))*chan_height
    #               
    #                    
    #                    strip_space_vertical = np.zeros([chan_height*chan_depth, chan_width])#preallocate space for output image with all channels concatenated
    #                    
    #                    for chan in range(0,chan_depth):
    #                        current_im = x_np[0,chan,:,:]
    #        #                current_im = (current_x/np.mean(current_x[:]))*127
    #        #                current_im[current_im>255] = 255
    #        #                current_im[current_im<0] = 0
    #                        current_im = np.uint8(current_im)
    #                        
    #                        start_x_ind = (chan)*chan_height
    #                        end_x_ind = (chan+1)*chan_height
    #                
    #                #        strip_space_horizontal(:, start_x_ind:end_x_ind) = uint8((a(:,:, global_frame_num)+20).*mask+50);
    #                        strip_space_vertical[start_x_ind:end_x_ind, :] = current_im
    #                
    #                        if chan>1:
    #                #            strip_space_horizontal(:, (start_x_ind-5):start_x_ind) = uint8(255);
    #                            strip_space_vertical[(start_x_ind-5):start_x_ind, :] = np.uint8(255);
    #                            
                            
        #            cv2.imwrite('Layer'+str(selected_layer)+'Activations'+'.jpg', strip_space_vertical)
    #                    for class_num in range(0, num_classes):
                    
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
                    
                    
                   
                        
                    
#                    np.save('CAM_glaucoma', weighted_classes_summed['class'+str(target+1)][:,:,:,target_idx])
    #                    weighted_class1 = x_np[0,:,:,:]*weights_np[0,:,None,None]
    #                    weighted_class2 = x_np[0,:,:,:]*weights_np[1,:,None,None]
    #                    weighted_class3 = x_np[0,:,:,:]*weights_np[2,:,None,None]
                        
    #                    weighted_class1_all[:,:,target_idx] = np.sum(weighted_class1,0)
    #                    weighted_class2_all[:,:,target_idx] = np.sum(weighted_class2,0)
    #                    weighted_class3_all[:,:,target_idx] = np.sum(weighted_class3,0)
            #            sio.savemat('Activations.mat', {'activation' :x_np} )
            #            sio.savemat('WeightedClass1.mat', {'class1w' :weighted_class1} )
            #            sio.savemat('WeightedClass2.mat', {'class2w' :weighted_class2} )
            #            break
#            for class_num in range(0, num_classes):
#                            
#                correct_ans_avg['run'+str(run_num)]['class'+str(class_num+1)] = np.mean(weighted_classes_summed['class'+str(class_num+1)][:,:,preds==class_num],2) 
    #        correct_ans_class1_avg = np.mean(weighted_class1_all[:,:,preds==0],2)
    #        correct_ans_class2_avg = np.mean(weighted_class2_all[:,:,preds==1],2)
    #        correct_ans_class3_avg = np.mean(weighted_class3_all[:,:,preds==2],2)
    #        
    #        correct_ans_class1_avg = correct_ans_class1_avg + abs(np.min(correct_ans_class1_avg))
    #        correct_ans_class2_avg = correct_ans_class2_avg + abs(np.min(correct_ans_class2_avg))
    #        correct_ans_class3_avg = correct_ans_class3_avg + abs(np.min(correct_ans_class3_avg))
    #            correct_ans_avg['class'+str(class_num+1)] = ((correct_ans_avg['class'+str(class_num+1)]/np.max(correct_ans_avg['class'+str(class_num+1)]))*255)
            
    #        correct_ans_class1_avg = ((correct_ans_class1_avg/np.max(correct_ans_class1_avg))*255).astype(np.uint8)
    #        correct_ans_class2_avg = ((correct_ans_class2_avg/np.max(correct_ans_class2_avg))*255).astype(np.uint8)
    #        correct_ans_class3_avg = ((correct_ans_class3_avg/np.max(correct_ans_class3_avg))*255).astype(np.uint8)
    #        
    #        save_settings = 'conf'+str(conf)+'_'+position_names[fix_pos]
    #        
    #        im_color = cv2.resize(cv2.applyColorMap(correct_ans_class1_avg, cv2.COLORMAP_JET), (224, 224))
    #        cv2.imwrite('correct_ans_'+save_settings+'_class1_avg.png',im_color)
    #        im_color = cv2.resize(cv2.applyColorMap(correct_ans_class2_avg, cv2.COLORMAP_JET), (224, 224))
    #        cv2.imwrite('correct_ans_'+save_settings+'_class2_avg.png',im_color)
    #        im_color = cv2.resize(cv2.applyColorMap(correct_ans_class3_avg, cv2.COLORMAP_JET), (224, 224))
    #        cv2.imwrite('correct_ans_'+save_settings+'_class3_avg.png',im_color)
            
    #        weighted_classes = {'class1': correct_ans_class1_avg, 'class2': correct_ans_class2_avg, 'class3': correct_ans_class3_avg }
            
#            sio.savemat('WeightedClasses_GenderTaskHN_mixedEmo_80_trainSet'+str(train_set_num)+'_testSet'+str(test_set_num)+'.mat', {'CAM': correct_ans_avg})


#num_classes = 2;
#map_width = 14;
#num_runs = 10;


#for train_set = 2
#    
#    for test_set = 1%length(position_names)
#
#        activation_maps = NaN(map_width, map_width, num_classes,num_runs);
#        
#        for run_num = 0:(num_runs-1)
#            for class_num = 1:2
#
#                file_path = fullfile('..', ['WeightedClasses_GenderTaskHN_mixedEmo_80_trainSet', num2str(train_set), '_testSet', num2str(test_set), '.mat']);
#
#                load(file_path)
#
#                activation_maps(:, :, class_num,run_num+1) = double(CAM.(['run', num2str(run_num)]).(['class', num2str(class_num)]));
#            end
#        end
#
#            activation_map = squeeze(nanmean(nanmean(activation_maps, 3),4));
#            activation_map = activation_map/max(max(activation_map));
#            
#            max_mm = max(max(activation_map));%+.000001;
#
#            load(fullfile('~', 'MEGA', 'Eckstein Lab', 'Faces','human_pics_paper'))
#            mm = [0 max_mm];
#            eb = 1.001;
#
#            refa=imresize(activation_map, [500 500], 'bilinear')-mm(1);                                                               %Set values relative to the bottom of the desired range
#            refa=refa/(mm(2)-mm(1));                                                    %Set values relative to the top of the desired range
#            leg=repmat(linspace(0,1,500),[20 1]);                                  %Create legend
#            refa=eb.^refa;                                                              %Exponentially scale range of image
#            leg=eb.^leg;                                                                %Exponentially scale range of legend
#            ima=mat2gray(refa,eb.^[0 1]);                                               %Create scaled intensity images for image and legend
#            imleg=mat2gray(leg,eb.^[0 1]);
#            ima2=gray2ind(ima,256);                                                     %Convert intensity images to indexed images (for alpha mapping)
#            imleg2=gray2ind(imleg,256);
#            ima=ind2rgb(ima2,jet(256));                                                 %Convert indexed images to RGB images with a jet colormap
#            imleg=ind2rgb(imleg2,jet(256));
#
#                hfig = figure();
#                % superimpose    
#            %        
#                imshow(repmat(uint8(mean(double(a(:,:,1)), 3))*2.*uint8(mask), [1 1 3])); hold on
#                h = imshow(ima); % show the fixation map
#                set( h, 'AlphaData', ima2.*uint8(mask)); % .5 transparency
#                hold off
#
#            %      colorbar('location', 'SouthOutside'); hold on
#            %     alphamap(h, .5);
#            %    
#            %     caxis([.25 .75]);
#            % 
#            % %show figure for colorbar
#            %  figure();
#            %     % superimpose    
#            % %        
#            %      g = imshow(imleg); % show the fixation map
#            %      set( g, 'AlphaData', imleg2); % .5 transparency
#            %    
#
#    end
#end    
#        
##    conv_output = x[0, selected_filter]
##    created_image = recreate_image(self.processed_image)
#            # Save image
##            if i % 5 == 0:
#                cv2.imwrite('../generated/layer_vis_l' + str(self.selected_layer) +
#                            '_f' + str(self.selected_filter) + '_iter'+str(i)+'.jpg',
#                            self.created_image)
