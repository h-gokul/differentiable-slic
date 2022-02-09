# Differentiable-slic
This project is a study on differentiable-slic, which uses superpixel sampling networks to obtain segmented optical flow

This repository is derived from the following research works

[Superpixel Sampling Networks](https://github.com/CYang0515/pytorch_ssn)
[Pyramidal Stereo Matching](https://github.com/JiaRenChang/PSMNet)
[FlowNet](https://github.com/ClementPinard/FlowNetPytorch)
[Recurrent All-Pairs Field Transform](https://github.com/princeton-vl/RAFT)
[Superpixel Segmentation using Fully Convolutional Networks](https://github.com/fuy34/superpixel_fcn)

# Motion Segmentation Task 

We will be using [Davis Dataset](https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip) for motion segmentation.

`pytorch_ssn` package has the differentiable slic + optical flow functions. 

The optical flow we use here is using a method called `RAFT`

We already finetuned the optical flow model with the simulated data from [here](https://drive.google.com/drive/folders/16V2-7NOEKJjsb3ChHGXy3AGudNjWGqA-?usp=sharing)


Differentiable slic
Refer the depthEstimation.ipynb notebook for step by step implementation details to obtain segmented optical flow
