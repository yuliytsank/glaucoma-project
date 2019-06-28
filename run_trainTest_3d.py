#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 23:12:21 2019

@author: yuliy
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import random
import numpy as np
import pdb
import copy
import os
import csv
import time
import cv2
import glob
import PIL
from scipy import ndimage
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split
#from imblearn.over_sampling import RandomOverSampler
import get_data_info
#import custom_transforms

epochs = 100
test_epochs = range(0, epochs, 2) 
save_epochs = range(0, epochs, 2) 

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=160, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=160, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default = epochs, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--train_rnn', type=int, default=1, metavar='N',
                    help='flag whether to train RNN (using saved CNN model), or CNN')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

args.lr = .001#lowered from .01, .00001 is too low
args.momentum = .9
#dropout_p = .5

#save_name = 'stats_test'
#save_name = 'stats_final_dataAug_crop_Vgg16Bn_lr-'+str(args.lr)+'_'+'m-'+str(args.momentum)+'_'+'drpOut-'+str(dropout_p)

classes = ['Normal', 'POAG']
trainSet_sizes = range(0,5)
#noise_values = [0, .01, .02]
noise_values = [0]
#contrast_values = [1,.95, .9 ]
contrast_values = [1]
transform = {}
transform['ang'] = 10
transform['shift']= 5
transform['noise'] = 5
transform = None

num_trainSet_sizes = len(trainSet_sizes)
num_noise_vals = len(noise_values)
num_con_vals = len(contrast_values)


num_runs = 5
num_trainSet_sizes = 5
stats = {}
stats['losses'] = {}
stats['losses']['train'] = np.zeros([epochs,num_runs,num_trainSet_sizes])
stats['losses']['val'] = np.zeros([epochs,num_runs,num_trainSet_sizes])
stats['losses']['test'] = np.zeros([epochs,num_runs,num_trainSet_sizes])
stats['perform'] = {}
stats['perform']['train'] = np.zeros([epochs,num_runs,num_trainSet_sizes])
stats['perform']['val'] = np.zeros([epochs, num_runs,num_trainSet_sizes])
stats['perform']['test'] = np.zeros([epochs, num_runs,num_trainSet_sizes])



stats['time'] = np.zeros([epochs,2])
stats['targets'] = {}
#stats['targets']['train']= np.full([epochs,num_runs,num_trainSet_sizes,1400],np.nan)#preallocate space for 1000 target values (way m)
stats['targets']['val']= np.full([epochs,num_runs,num_trainSet_sizes,250],np.nan)#preallocate space for 1000 target values (way m)
stats['targets']['test']= np.full([epochs,num_runs,num_trainSet_sizes,250],np.nan)#preallocate space for 1000 target values (way m)
stats['preds'] = {}
#stats['preds']['train']= np.full([epochs,num_runs,num_trainSet_sizes,1400],np.nan)#preallocate space for 1000 output values (way m)
stats['preds']['val']= np.full([epochs,num_runs,num_trainSet_sizes,250],np.nan)#preallocate space for 1000 output values (way m)
stats['preds']['test']= np.full([epochs,num_runs,num_trainSet_sizes,250],np.nan)#preallocate space for 1000 output values (way m)
#stats['metrics']['auc'] = 
stats['probs']={}
#stats['probs']['train']= np.full([epochs,num_runs,num_trainSet_sizes,1400],np.nan)
stats['probs']['val']= np.full([epochs,num_runs,num_trainSet_sizes,250],np.nan)
stats['probs']['test']= np.full([epochs,num_runs,num_trainSet_sizes,250],np.nan)

stats['paths'] = {}
#stats['paths']['train']= np.empty([epochs,num_runs,num_trainSet_sizes,1400], dtype="S35")
stats['paths']['val']= np.empty([epochs,num_runs,num_trainSet_sizes,250], dtype="S35")
stats['paths']['test']= np.empty([epochs,num_runs,num_trainSet_sizes,250], dtype="S35")

stats['auc'] = {}
stats['auc']['train'] = np.full([epochs,num_runs,num_trainSet_sizes],np.nan)
stats['auc']['val'] = np.full([epochs,num_runs,num_trainSet_sizes],np.nan)
stats['auc']['test'] = np.full([epochs,num_runs,num_trainSet_sizes],np.nan)



torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    
    
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

#class OCT_Folder(torch.utils.data.Dataset):
#    def __init__(self, root, loader, extensions=None, transform=None, target_transform=None, is_valid_file=None):
#        self.data = torch.from_numpy(data[None,:,:,:])
#        self.target = torch.from_numpy(target)
#        self.transform = transform
#        
#    def __getitem__(self, index):
#        x = self.data[index]
#        y = self.target[index]
#        
#        if self.transform:
#            x = self.transform(x)
#            
#        return x, y
#    
#    def __len__(self):
#        return len(self.data)
extract_dir = os.path.expanduser(os.path.join('~', 'Desktop', 'InsightProjectData', 'NP_Volumes', 'subsampled_all'))
data_info = get_data_info.get_data_info(path_data_root = extract_dir)
data_info_train = dict(data_info)
data_info_test = dict(data_info)


data_info_train['paths'], data_info_test['paths'], data_info_train['targets'], data_info_test['targets'] = train_test_split(data_info['paths'], data_info['targets'], test_size=0.1)

#split into cross validation sets
kf = KFold(n_splits=num_runs)



class OCT_Folder(datasets.DatasetFolder):
    
    def __init__(self, root, loader = datasets.folder.default_loader, extensions='.npy', transform=None, target_transform=None, is_valid_file=None, data_info=data_info):
        super(OCT_Folder, self).__init__(root, loader, extensions=extensions)
        self.transform = transform
#        self.target_transform = target_transform
#        classes, class_to_idx = datasets.folder.find_classes(self.root)
#        samples = datasets.folder.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
#        if len(samples) == 0:
#            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
#                                "Supported extensions are: " + ",".join(extensions)))
        
        samples = [[a,b] for (a,b) in zip(data_info['paths'], data_info['targets'])]
        self.classes = data_info['classes']
        self.class_to_idx = data_info['class_to_idx']
        self.samples = samples
        self.targets = [s[1] for s in samples]
#        self.extensions = extensions
#
#        self.classes = classes
#        self.class_to_idx = class_to_idx
#        self.samples = samples
#        self.targets = [s[1] for s in samples]
        
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
#        sample = self.numpy3d_loader(path)
        sample = np.load(path)
        
        if self.transform is not None:
            
            ang = self.transform['ang']
            shift_val = self.transform['shift']
            noise_val = self.transform['noise']
            
            rand_ang1, rand_ang2  = np.random.randint(-ang,ang, size=(2) )
            sample = ndimage.rotate(sample, rand_ang1, axes=(1,0), mode='nearest', reshape = False)
            sample = ndimage.rotate(sample, rand_ang2, axes=(2,1), mode='nearest', reshape = False)
        
            shift_vals = np.random.randint(-shift_val, shift_val, size=(3))
            sample = ndimage.shift(sample, shift_vals, mode='nearest')
            sample = sample+np.random.normal(0, scale = noise_val, size = sample.shape)
#            sample = self.transform(sample)
        
        sample = torch.from_numpy(sample[None,:,:,:]).float()
        self.target = torch.tensor([[target]])
        
        path = os.path.split(path)[-1]
        
        return path, sample, target
    
  


    
#class ToTensor3D(transforms.ToTensor):
#    
#    def __call__(self, pic):
#        pic = pic[None,:,:,:]#add channel dimension
#        vol_tensor = torch.from_numpy(pic)
#        return vol_tensor
           
#    def randomRotation3D(vol,d1,d2,d3):
#        hello = 'hello'
#        return hello
    
def addGaussNoise_andContrast(data, std_val,con):
    n_ims, n_chans, h,w = data.size()
    data = data.float()*con+torch.normal(0., std = torch.ones(n_ims,1,h,w)*std_val).repeat(1,3,1,1)
#    temp[temp<0] = 0
#    temp[temp>255] = 255    
#    data = temp.byte()
    return data       
#    train_loader.dataset.train_data = torch.clamp(train_loader.dataset.train_data, max=255) # cmin
#    train_loader.dataset.train_data = torch.clamp(train_loader.dataset.train_data, min=0) # cmax

def randomize_labels(train_loader, proportion):
    s = train_loader.dataset.train_labels.size()
    num_labels = int(round(int(s[0])*proportion))
    temp = range(0,len(train_loader.dataset.train_labels))
    random.shuffle(temp)
    temp2 = temp[0:num_labels]
    random.shuffle(temp2)
    temp3 = copy.copy(temp)
    temp3[0:num_labels] = temp2
    train_loader.dataset.train_labels[torch.LongTensor(temp[0:num_labels])] = train_loader.dataset.train_labels[torch.LongTensor(temp3[0:num_labels])]
    return train_loader


import custom3d_GAP_glaucoma

def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    
    targets_all = np.full([1400],np.nan)
    probs_all = np.full([1400],np.nan)
    
    for batch_idx, (path, data, target) in enumerate(train_loader):
        
#            data = addGaussNoise_andContrast(data,.01,1.)
        
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
#        pdb.set_trace()
        output = model(data)
        probs =  np.around(softmax(output)[:,1].detach().cpu().numpy(),2)
#        loss = F.nll_loss(output, target)
        loss = criterion(output, target)
        train_loss += loss.data[0]
        pred = output.data.max(1)[1].cpu().numpy() # get the index of the max log-probability
#        correct += pred.eq(target.data).cpu().sum()
        targets = target.data.cpu().numpy()
        correct += (targets==pred).sum()
        
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(' RunNum: ' +str(run_num)+ ' TrainSetSize: ' +str(trainSet_size_num)+ ' Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
        
        start_ind = int(np.argwhere(np.isnan(targets_all))[0])
        end_ind = start_ind +len(targets)
        targets_all[start_ind:end_ind]=targets
        probs_all[start_ind:end_ind]=probs
#        stats['targets']['train'][epoch, run_num,trainSet_size_num,start_ind:end_ind] = targets
#        stats['preds']['train'][epoch, run_num,trainSet_size_num,start_ind:end_ind] = pred.cpu().numpy()
#        stats['probs']['train'][epoch, run_num,trainSet_size_num,start_ind:end_ind] = np.around(probs.detach().cpu().numpy(),2)
#        stats['paths']['train'][epoch, run_num,trainSet_size_num,start_ind:end_ind] = path
    fpr, tpr, thresholds = metrics.roc_curve(targets_all[0:end_ind], probs_all[0:end_ind], pos_label=1)
    auc_score = metrics.auc(fpr, tpr)
    stats['auc']['train'][epoch, run_num,trainSet_size_num] = auc_score
    train_loss /=len(train_loader)  
    correct_prop = np.float(correct) / np.float(len(train_loader.dataset))
    stats['perform']['train'][epoch, run_num,trainSet_size_num] = correct_prop
    stats['losses']['train'][epoch, run_num,trainSet_size_num] = train_loss
#    np.save('train_losses_vgg16_bn', train_losses)
#preallocate space for confusion matrix

           
softmax = nn.Softmax()


def test(epoch, loader_type):
#    confusion_sums = np.zeros([100,100])
#    class_trials = np.zeros([100,100])
#    probs_list = np.empty((len(test_ids),100))
#    scores_list = np.empty((len(test_ids),100))
#    for test_set_size_num, test_set_size in enumerate(trainSet_sizes):
       
    if loader_type == 'val':
        loader = val_loader
    else:
        loader = test_loader
        
        
    model.eval()
    
   
        
    test_loss = 0
    correct = 0
    test_batch = 0
    for batch_num, (path, data, target) in enumerate(loader):
        
#                    data = addGaussNoise_andContrast(data,noise_values[noise_ind], contrast_values[contrast_ind])
        
        test_batch+=1
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
#        scores = output
        probs =  np.around(softmax(output)[:,1].detach().cpu().numpy(),2)#glaucoma present is target 1
#        probs_list[range(args.test_batch_size*(test_batch-1),args.test_batch_size*test_batch),:] = probs.data.cpu().numpy()
#        scores_list[range(args.test_batch_size*(test_batch-1),args.test_batch_size*test_batch),:] = scores.data.cpu().numpy()
#        test_loss += F.nll_loss(output, target).data[0]
        test_loss += criterion(output, target).data[0]
        pred = output.data.max(1)[1].cpu().numpy() # get the index of the max log-probability
        
#        correct += pred.eq(target.data).cpu().sum()
#        targets = target.data.cpu().numpy()
        targets = target.data.cpu().numpy()
        correct += (targets==pred).sum()
                    
        start_ind = int(np.argwhere(np.isnan(stats['targets'][loader_type][epoch, run_num,trainSet_size_num,:]))[0])
        end_ind = start_ind +len(targets)
        stats['targets'][loader_type][epoch, run_num,trainSet_size_num,start_ind:end_ind] = targets
        stats['preds'][loader_type][epoch, run_num,trainSet_size_num,start_ind:end_ind] = pred
        stats['probs'][loader_type][epoch, run_num,trainSet_size_num,start_ind:end_ind] = probs
        
               
        if (trainSet_size_num == (num_trainSet_sizes-1))&(run_num==0)&(epoch>30):
            stats['paths'][loader_type][epoch, run_num,trainSet_size_num,start_ind:end_ind] = path
    
    targets_all = stats['targets'][loader_type][epoch, run_num,trainSet_size_num,0:end_ind]
    probs_all = stats['probs'][loader_type][epoch, run_num,trainSet_size_num,0:end_ind]
    fpr, tpr, thresholds = metrics.roc_curve(targets_all, probs_all, pos_label=1)
    auc_score = metrics.auc(fpr, tpr)
    stats['auc'][loader_type][epoch, run_num,trainSet_size_num] = auc_score
    
    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\n' + ' RunNum: ' +str(run_num)+ ' TrainSetSize: ' +str(trainSet_size_num)+ ' Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(loader.dataset),
        100. * correct / len(loader.dataset)))
    correct_prop = np.float(correct) / np.float(len(loader.dataset))
    
    stats['perform'][loader_type][epoch, run_num,trainSet_size_num] = correct_prop
    stats['losses'][loader_type][epoch, run_num,trainSet_size_num] = test_loss
    
                #    stats['time'][epoch-1] = time.time()-start 
    np.save(save_name, stats) 

    return correct_prop

batch_size = args.batch_size


for run_num, (train_index, val_index) in enumerate(kf.split(data_info_train['targets'])):
    
#    ros = RandomOverSampler(random_state=42)
    
    data_info_fold_train = dict(data_info_train)
    data_info_fold_val = dict(data_info_train)
    
#    oversample only for two classes
    targets_fold = np.asarray(data_info_train['targets'])[train_index].tolist()
    class0_count = targets_fold.count(0)
    class1_count = targets_fold.count(1)
    class_counts = [class0_count, class1_count]
    
    
    oversample_all_times = int(np.floor(class1_count/class0_count))
    oversample_remainder = class1_count % class0_count
        
    less_num_inds = train_index[np.where(np.asarray(targets_fold)==0)]
    less_num_inds_size = len(less_num_inds)
    train_index_updated = np.concatenate((train_index, np.repeat(less_num_inds , (oversample_all_times-1))))
    train_index_updated = np.concatenate((train_index_updated, less_num_inds[np.random.randint(less_num_inds_size, size = (oversample_remainder))]))
    np.random.shuffle(train_index_updated)
    
#    data_info_fold_train['paths'] = np.asarray(data_info_train['paths'])[train_index_updated].tolist()
#    data_info_fold_train['targets'] = np.asarray(data_info_train['targets'])[train_index_updated].tolist()
    
#    X_res, y_res = ros.fit_resample(data_info_fold_train['paths'], data_info_fold_train['targets'])
    
#    data_info_fold_train['paths'] = X_res.tolist()
#    data_info_fold_train['targets'] = y_res.tolist()
    
    data_info_fold_val['paths'] = np.asarray(data_info_train['paths'])[val_index].tolist()
    data_info_fold_val['targets'] = np.asarray(data_info_train['targets'])[val_index].tolist()
       
    
    num_train_samples_fold = np.size(train_index_updated)
    subfold_size = np.round(num_train_samples_fold/num_trainSet_sizes)
    subfold_end_inds = range(subfold_size, subfold_size*(num_trainSet_sizes), subfold_size)
    subfold_end_inds = subfold_end_inds[0:(num_trainSet_sizes-1)]
    subfold_end_inds.append(num_train_samples_fold)
    
    for trainSet_size_num, subfold_end_ind in enumerate(subfold_end_inds):
 
        data_info_fold_train['paths'] = np.asarray(data_info_train['paths'])[train_index_updated[0:subfold_end_ind]].tolist()
        data_info_fold_train['targets'] = np.asarray(data_info_train['targets'])[train_index_updated[0:subfold_end_ind]].tolist()
        
        model = custom3d_GAP_glaucoma.Glaucoma3d(num_classes = 2)
        if args.cuda:
            model.cuda()
#            model.features = torch.nn.DataParallel(model.features)#only needed for multiple GPUs and makes loading/saving models more complicated
            criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    
        save_name = 'custom_3d_stats_all_'+'glaucoma'#+ base_foveation_dir
#        if pos == 1:
#            save_name = 'stats_eyes'
        path_to_train_data = os.path.expanduser(os.path.join('~', 'Desktop', 'InsightProjectData', 'NP_Volumes', 'training'))
        path_to_test_data_root = os.path.expanduser(os.path.join('~', 'Desktop', 'InsightProjectData', 'NP_Volumes','validation'))
        train_loader = torch.utils.data.DataLoader(
                OCT_Folder(path_to_train_data, data_info = data_info_fold_train, transform = transform),
                batch_size=args.batch_size, shuffle=True, **kwargs)
#        train_loader = torch.utils.data.DataLoader(
#            OCT_Folder(path_to_train_data, data_info = data_info_fold_train, transform = transform),
#            batch_size=args.batch_size, shuffle=True, **kwargs)
        
        path_to_test_data = os.path.join(path_to_test_data_root)#,test_set_size)     
        val_loader = torch.utils.data.DataLoader(
            OCT_Folder(path_to_test_data, data_info = data_info_fold_val),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            OCT_Folder(path_to_test_data, data_info = data_info_test),
            batch_size=args.batch_size, shuffle=True, **kwargs)
      
        for epoch in range(0, args.epochs):
   
            
            train(epoch)
            if epoch in test_epochs:
                test(epoch, 'val')
                test(epoch, 'test')

#                if (run_num==0)&(epoch == (epochs-1)):
                path_saved_model = os.path.join('.', 'model_glaucoma_setSize'+ str(trainSet_size_num) +'_epoch_'+str(epoch)+'_run_' +str(run_num))
                print('Saving model...')
                torch.save(model.state_dict(), path_saved_model)