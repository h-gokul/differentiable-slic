This repository contains a detailed study on superpixels, as a part of the course CMSC828i, at the University of Maryland.

# Segmentation with superpixels

```Kmeans-SLIC``` notebook contains code for of Kmeans image clustering and Simple Linear Iterative Clustering for superpixel generation. The implementations are built from scratch using python

```Segmentation``` notebook contains code for a image segmentation pipeline that classifies superpixel patches using a VGG-19 backbone, and assembles the predictions to form a segmentation map. The results of these pipline are further improved by incorporating and zoom out features.


# Differentiable SuperPixel networks
This project is a study on differentiable-slic, where we integrate two methods for superpixel generation to integrate in any deep learning architecture. We used implementations of [SuperPixel Sampling Networks](https://github.com/CYang0515/pytorch_ssn) and [Fully convolutional Superpixel Networks](https://github.com/fuy34/superpixel_fcn)

## Link to video PPT
https://drive.google.com/file/d/1P7TwEfejR2gMQMzcnIKwnp3awxkAjLap/view?usp=sharing

## Motion Segmentation Task 

- This folder contains the tests we ran for Optical flow. Check `OpticalFlow.ipynb` for flownetS's results and our test on enforcing smoothness constraints using superpixels.

Refer `Segmentation.ipynb` for tests on Video Segmentation Architecture. 

We will be using [Davis Dataset](https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip) for motion segmentation.

`pytorch_ssn` package has the differentiable slic + optical flow functions. 

The optical flow we use here is using a method called `RAFT`

We also tried to finetune the optical flow model with the simulator engine [Airsim](https://github.com/microsoft/AirSim). A subset of the data is [here](https://drive.google.com/drive/folders/16V2-7NOEKJjsb3ChHGXy3AGudNjWGqA-?usp=sharing)

Even though we want to train the model end to end, due to resource and time constraints, we will choose to pre generate the superpixels and optical flow from the corresponding networks. `slicgenerator.py` generates the SuperpixelSamplingNetwork's conv-features and saves them. This network is fairly generalizable, hence it was not finetuned.  `flowgenerator.py` generates optical flow and saves them. Load these two, along with images and groundtruth in the dataloader `DavisMoSegLoader` and train the UNet network. Refer `train_unet.py` for the train and validation functions.

Import the checkpoints and test-dataset from the folder titled `motiontaskfiles` from this [folder](https://drive.google.com/drive/folders/1MgfStqB3Nx0tfnJRXc6n0vvoWoEt6N8H?usp=sharing). The video outputs for optical flow are also present in the same link.


## Disparity Task

We experimented the paper "Superpixel Segmentation using fully connected convolutional networks" and thier application of using super pixel network for upsampling stereo images. We achieved very good results compared to the baseline. You can find the results of the baseline [here](https://drive.google.com/drive/folders/1MgfStqB3Nx0tfnJRXc6n0vvoWoEt6N8H?usp=sharing) Import the checkpoints and test-dataset from the folder titled `disptaskfiles` in the same link.

import the superpixel fcn's pretrained checkpoint from [here](https://drive.google.com/file/d/1c5ZvzK2qiY6FYXaUayHUItbH2rrOXPcK/view?usp=sharing) and paste in `dispTask/superpixelnet/pretrain_ckpt/` 

## Classification Task
The code for image classification can be found in the pytorch-cifar folder. Nothing has been changed from default traiing except for model instantiation in main.py. All work done outside main.py was done in models folder. In the models folder, convmixer.py defines a standard Convmixer from the Patches are All You Need Paper [here](https://github.com/tmp-iclr/convmixer). convmixerSSN.py defines the convmixer with superpixel sampling for the patch embeddings. The patch is attended according to the soft pixel mappings produced by ssn, then a linear embedding over the patch produces the patch representation. In the ssn_pytorch_joshuasmith folder is the code of differentiable superpixel segmentation. The model.py and lib/ssn.py files have been modified to run 5-10x faster on cifar data by replacing the sparse affinity matrix calculation with regionwise stacking and dense matrix operations.

Future Works
[TODO]  : Follow the same routine, but train the U-Net for [multi-object video segmentation using Davis Dataset](https://davischallenge.org/davis2017/code.html#unsupervised)
[TODO]  : Motion Segmentation Pipeline a bayesian learning framework instead of relying on Residual UNets.
  
