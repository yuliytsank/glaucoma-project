
import numpy as np
from sklearn import metrics

import matplotlib.pyplot as plt

num_epochs = 100
num_runs = 5

stats_all = np.load('custom_3d_stats_all_glaucoma.npy').item()

#for train_set in train_sets:
num_trainSetSizes = 5
test_set = 4

means_and_stds = np.empty((2,num_trainSetSizes))*np.nan

min_ind_epochs = np.empty((num_trainSetSizes))*np.nan
#    for test_set in test_sets:
#        [test_fix_pos, epoch-1, run_num,train_fix_pos,noise_ind, contrast_ind]
auc_scores = np.full((num_runs, num_trainSetSizes),np.nan)
auc_scores_curve_train = np.full((num_runs,num_epochs),np.nan)
auc_scores_curve_val = np.full((num_runs,num_epochs),np.nan)
auc_scores_curve_test = np.full((num_runs,num_epochs),np.nan)

losses_curve_train = np.full((num_runs,num_epochs),np.nan)
losses_curve_val = np.full((num_runs,num_epochs),np.nan)
losses_curve_test = np.full((num_runs,num_epochs),np.nan)


for train_setSize in range(0,num_trainSetSizes):
    
    current_mean_train_losses = np.mean(stats_all['losses']['train'][:,:,train_setSize][:,0:num_epochs], 1)
    current_std_train_losses = np.std(stats_all['losses']['train'][:,:,train_setSize][:,0:num_epochs], 1)/np.sqrt(num_epochs)
#        
    current_mean_val_losses = np.mean(stats_all['losses']['val'][:,:,train_setSize][:,0:num_epochs], 1)
    current_std_val_losses = np.std(stats_all['losses']['val'][:,:,train_setSize][:,0:num_epochs], 1)/np.sqrt(num_epochs)
    
    current_mean_test_losses = np.mean(stats_all['losses']['test'][:,:,train_setSize][:,0:num_epochs], 1)
    current_std_test_losses = np.std(stats_all['losses']['test'][:,:,train_setSize][:,0:num_epochs], 1)/np.sqrt(num_epochs)
            
    min_ind = np.nanargmax(current_mean_val_losses)
    
    min_ind_epochs[train_setSize] = min_ind
    
    means_and_stds[0,train_setSize] = current_mean_test_losses[min_ind]
    means_and_stds[1,train_setSize] = current_std_test_losses[min_ind]
    
    for cv_fold in range(0,num_runs):
    
    
        
#        current_mean_train = np.mean(stats_all['perform']['train'][:,:,train_setSize][:,0:num_epochs], 1)
#        current_std_train = np.std(stats_all['perform']['train'][:,:,train_setSize][:,0:num_epochs], 1)/np.sqrt(num_epochs)
##        
#        current_mean_val = np.mean(stats_all['perform']['val'][:,:,train_setSize][:,0:num_epochs], 1)
#        current_std_val = np.std(stats_all['perform']['val'][:,:,train_setSize][:,0:num_epochs], 1)/np.sqrt(num_epochs)
#        
#        current_mean_test = np.mean(stats_all['perform']['test'][:,:,train_setSize][:,0:num_epochs], 1)
#        current_std_test = np.std(stats_all['perform']['test'][:,:,train_setSize][:,0:num_epochs], 1)/np.sqrt(num_epochs)
        
        
        
#        print(means_and_stds)
        end_ind = int(np.argwhere(np.isnan(stats_all['preds']['test'][min_ind, 0, train_setSize,:]))[0])-1
        predictions = stats_all['preds']['test'][min_ind, cv_fold, train_setSize, 0:end_ind]
        targets = stats_all['targets']['test'][min_ind, cv_fold, train_setSize, 0:end_ind]
        probs = stats_all['probs']['test'][min_ind, cv_fold, train_setSize, 0:end_ind]
        
        fpr, tpr, thresholds = metrics.roc_curve(targets, probs, pos_label=1)
        auc_score = metrics.auc(fpr, tpr)
        auc_scores[cv_fold, train_setSize] = auc_score
        f2beta_score = metrics.fbeta_score(targets, predictions, beta=2)
        
        if (train_setSize == (num_trainSetSizes-1)):
            for epoch_num in range(0,num_epochs,2):
                    auc_scores_curve_train[cv_fold, epoch_num] = stats_all['auc']['train'][epoch_num, cv_fold, train_setSize]
                    auc_scores_curve_val[cv_fold, epoch_num] = stats_all['auc']['val'][epoch_num, cv_fold, train_setSize]
                    auc_scores_curve_test[cv_fold, epoch_num] = stats_all['auc']['test'][epoch_num, cv_fold, train_setSize]
                    
                    losses_curve_train[cv_fold, epoch_num] = stats_all['losses']['train'][epoch_num, cv_fold, train_setSize]
                    losses_curve_val[cv_fold, epoch_num] = stats_all['losses']['val'][epoch_num, cv_fold, train_setSize]
                    losses_curve_test[cv_fold, epoch_num] = stats_all['losses']['test'][epoch_num, cv_fold, train_setSize]
        
        if (cv_fold ==0)&(train_setSize == (num_trainSetSizes-1)):
            plt.figure()
            lw = 5
            plt.rcParams.update({'font.size': 16})
            plt.plot(fpr, tpr, color='darkorange',
                     lw=lw, label='ROC curve (area = %0.2f)' % auc_score)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.show()
            
#plot performance as a function of training set size
auc_scores_mean = np.mean(auc_scores, 0)
auc_scores_std = np.std(auc_scores,0)
plt.figure()
lw = 5
plt.rcParams.update({'font.size': 16})
#plt.plot(range(0,880, 176), auc_scores_mean, color='darkorange',
#         lw=lw)
plt.errorbar(range(175,880, 176), auc_scores_mean,  yerr=auc_scores_std, markersize = 10,
             fmt = 'go-', ecolor = 'black', lw=lw, elinewidth = 2, capsize=5)
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#plt.xlim([0.0, 1.0])
plt.ylim([0.7, 1.0])
plt.xlabel('Training Set Size')
plt.ylabel('AUC Score')
plt.title('Effects of Training Set Size')
plt.legend(loc="lower right")
plt.show()
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#plt.xlim([0.0, 1.0])

#plot auc performance for train and validation sets as a function of epoch
plt.figure()
lw = 5
plt.rcParams.update({'font.size': 16})
#plt.plot(range(0,880, 176), auc_scores_mean, color='darkorange',
#         lw=lw)
auc_scores_curve_train_mean = np.mean(auc_scores_curve_train,0)
auc_scores_curve_val_mean = np.mean(auc_scores_curve_val,0)

auc_scores_curve_train_std = np.std(auc_scores_curve_train,0)
auc_scores_curve_val_std = np.std(auc_scores_curve_val,0)

not_nan_inds = np.isfinite(auc_scores_curve_val_mean)
all_epochs = np.array(range(0,num_epochs))
x_axis_vals = all_epochs[not_nan_inds]

plt.errorbar(x_axis_vals, auc_scores_curve_train_mean[not_nan_inds],  yerr=auc_scores_curve_train_std[not_nan_inds], markersize = 10,
             fmt = 'b-', ecolor = 'black', lw=lw, elinewidth = 2, capsize=5)

plt.errorbar(x_axis_vals, auc_scores_curve_val_mean[not_nan_inds],  yerr=auc_scores_curve_val_std[not_nan_inds], markersize = 10,
             fmt = 'g-', ecolor = 'black', lw=lw, elinewidth = 2, capsize=5)
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#plt.xlim([0.0, 1.0])
plt.ylim([0.7, 1.0])
plt.xlabel('Epoch Number')
plt.ylabel('AUC Score')
plt.title('Effects of Training Set Size')
plt.legend(loc="lower right")
plt.show()

#plot losses performance for train and validation sets as a function of epoch
plt.figure()
lw = 5
plt.rcParams.update({'font.size': 16})
#plt.plot(range(0,880, 176), auc_scores_mean, color='darkorange',
#         lw=lw)
losses_curve_train_mean = np.mean(losses_curve_train,0)
losses_curve_val_mean = np.mean(losses_curve_val,0)

losses_curve_train_std = np.std(losses_curve_train,0)
losses_curve_val_std = np.std(losses_curve_val,0)

not_nan_inds = np.isfinite(losses_curve_val_mean)
all_epochs = np.array(range(0,num_epochs))
x_axis_vals = all_epochs[not_nan_inds]

plt.errorbar(x_axis_vals, losses_curve_train_mean[not_nan_inds],  yerr=losses_curve_train_std[not_nan_inds], markersize = 10,
             fmt = 'b-', ecolor = 'black', lw=lw, elinewidth = 2, capsize=5)

plt.errorbar(x_axis_vals, losses_curve_val_mean[not_nan_inds],  yerr=losses_curve_val_std[not_nan_inds], markersize = 10,
             fmt = 'g-', ecolor = 'black', lw=lw, elinewidth = 2, capsize=5)
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#plt.xlim([0.0, 1.0])
plt.ylim([0.2, 1.4])
plt.xlabel('Epoch Number')
plt.ylabel('AUC Score')
plt.title('Effects of Training Set Size')
plt.legend(loc="lower right")
plt.show()


