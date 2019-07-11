# Glaucoma Detection Automation Project Using a 3D CNN

This was a project done at Insight Data Science in the summer of 2019 as a proof of concept of automated glaucoma detection in developing world regions from low-resolution Ocular Coherence Tomography (OCT) images. See www.glaucomaproject.xyz for a web app that I built that takes a low-resolution OCT volume as an input, and outputs a glaucoma probability score using a trained 3D CNN model. 

OCT volume reference:  
<img src="/Images/Reference/OCT_vol_reference.png" height="70%" width="70%">

## Requirements

-"requirements.txt" contains a long list of packages in the conda environment I was using with Python 2.7

-the main ones needed for this part of the project were pytorch, torchvision, scikit-learn 

## Training and Testing
```
python runTrainTest.py --help
```

```
optional arguments:
  -h, --help         show this help message and exit
  --batch-size N     input batch size for training (default: 160)
  --epochs N         number of epochs to train (default: 100)
  --lr LR            learning rate (default: 0.001)
  --momentum M       SGD momentum (default: 0.9)
  --no-cuda          enables CUDA training
  --seed S           random seed (default: 1)
  --log-interval LI  how many batches to wait before logging training status
  --extract-dir ED   directory where training data is located (default:
                     "./Images/Data/NP_Volumes/subsampled_all"
  --save-dir SD      directory to save models (default: "models")

```

This script trains on 3d OCT volume data saved in the "Images/Data/NP_Volumes/subsampled_all" directory. The data is split into 5 folds for cross-validation and separate training runs are done for subsets of the training data (i.e. 1/5 of the data, 2/5 of the data,... 5/5 of the data) to observe effects of the training data size. Separate models (parameter sets) are saved in the "Models" directory for each cross-validation fold as well as for each training set size and for every other epoch within each training run, for a total of (5 folds x 5 train set sizes x 50 epochs = 1250 models). However, since the network is small, each model takes up only about 400kb of space. 

This script also validates each training run on every other epoch and records the results of loss, performance, probabilities (outputs of softmax function), and class targets shown on each trial, in 'custom_3d_stats_all_glaucoma.npy' for further analysis. 

## Network Details

- the network architecture is contained in 'utils/custom3d_GAP_glaucoma.py'

-the network I built was based off of the one in this medical imaging paper: https://arxiv.org/abs/1807.04855

-the image below shows the original network at the top and my customization below it in order to fit low-resolution data to simulate the output of a less expensive OCT scanner

<img src="/Images/Reference/NetworkArchitecture.png" height="7=100%" width="100%">

## Performance Analysis

Area Under Curve (AUC) Learning as a function of Epoch Number:  
<img src="/Images/Reference/Learning_Curves_AUC.png" height="70%" width="70%">

Cross Entropy Loss as a function of Epoch Number:  
<img src="/Images/Reference/Learning_Curves_Loss.png" height="70%" width="70%">

Effects of Training Set Size on Test Set Performance:  
<img src="/Images/Reference/TrainingSetSizeEffects.png" height="70%" width="70%">

Receiver Operating Characteristic (ROC) Curve When Using Full Training Set for One of the Cross Validation Runs:  
<img src="/Images/Reference/ROC_Glaucoma.png" height="70%" width="70%">

## Script for Plotting Performance Analysis

- "analyze_performance.py" in the "utils" directory is used for plotting the performance curves above

## Script for Computing Class Activation Maps (CAMs)

- "precompute_CAMs.py" calls "visualize_slice_CAM_glaucoma3d.py" to compute CAMs for 3 enface slices and 3 cross-section slices of the volume (see OCT reference image at the top for a visuzalization of where these slices come from) for all samples when they were part of the validation set in each cross-validation fold

- see www.glaucomaproject.xyz for a web app that I built with examples of CAM outputs for different diagnostic cases

Examples of CAMs:  
<img src="/Images/Reference/WebApp_CAMs.png" height="70%" width="70%">

- see http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf for the paper that first introduced CAMs
