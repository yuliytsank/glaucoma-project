# glaucoma-project

This was project done at Insight Data Science in the summer of 2019 as a proof of concept of automated glaucoma detection from low- resolution Ocular Coherence Tomography (OCT) images. 

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

This script trains on 3d OCT volume data saved in the 'Images/Data/NP_Volumes/subsampled_all' directory. The data is split into 5 folds for cross-validation and separate training runs are done for subsets of the training data (i.e. 1/5 of the data, 2/5 of the data,... 5/5 of the data) to observe effects of the training data size. Separate models (parameter sets) are saved for each cross-validation fold as well as for each training set size and for every other epoch within each trining run, for a total of (5 folds x 5 train set sizes x 50 epochs = 1250 models). However, since the network is small, each model takes up only about 400kb of space. 

This script also validates each training run on every other epoch and records the results of loss, performance, probabilities (outputs of softmax function), and class targets shown on each trial, in 'custom_3d_stats_all_glaucoma.npy' for further analysis. 
