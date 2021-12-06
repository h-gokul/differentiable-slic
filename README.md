# Differentiable-slic
This project is a study on differentiable-slic, which uses superpixel sampling networks to obtain segmented optical flow

## Motion Segmentation Task 

We will be using [Davis Dataset](https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip) for motion segmentation.

`pytorch_ssn` package has the differentiable slic + optical flow functions. 

The optical flow we use here is using a method called `RAFT`

We already finetuned the optical flow model with the simulated data from [here](https://drive.google.com/drive/folders/16V2-7NOEKJjsb3ChHGXy3AGudNjWGqA-?usp=sharing)

Even though we want to train the model end to end, due to resource and time constraints, we will choose to pre generate the superpixels and optical flow from the corresponding networks. `slicgenerator.py` generates the SuperpixelSamplingNetwork's conv-features and saves them. This network is fairly generalizable, hence it was not finetuned.  `flowgenerator.py` generates optical flow and saves them. Load these two, along with images and groundtruth in the dataloader `DavisMoSegLoader` 
and train the UNet network. Refer `train_unet.py` for the train and validation functions.

Additionally

Differentiable slic
Refer the depthEstimation.ipynb notebook for step by step implementation details to obtain optical flow and compute depth using airsim data which is present [here](https://drive.google.com/drive/folders/16V2-7NOEKJjsb3ChHGXy3AGudNjWGqA-?usp=sharing) 



