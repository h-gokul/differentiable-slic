# Differentiable-slic
This project is a study on differentiable-slic, which uses superpixel sampling networks to obtain segmented optical flow

## Motion Segmentation Task 

- This repo also contains the tests we ran for Optical flow. Check `OpticalFlow.ipynb` for flownetS's results and our test on enforcing smoothness constraints using superpixels.

Refer `Segmentation.ipynb` for tests on Video Segmentation Architecture. 

We will be using [Davis Dataset](https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip) for motion segmentation.

`pytorch_ssn` package has the differentiable slic + optical flow functions. 

The optical flow we use here is using a method called `RAFT`

We already finetuned the optical flow model with the simulated data from [here](https://drive.google.com/drive/folders/16V2-7NOEKJjsb3ChHGXy3AGudNjWGqA-?usp=sharing)

Even though we want to train the model end to end, due to resource and time constraints, we will choose to pre generate the superpixels and optical flow from the corresponding networks. `slicgenerator.py` generates the SuperpixelSamplingNetwork's conv-features and saves them. This network is fairly generalizable, hence it was not finetuned.  `flowgenerator.py` generates optical flow and saves them. Load these two, along with images and groundtruth in the dataloader `DavisMoSegLoader` and train the UNet network. Refer `train_unet.py` for the train and validation functions.

Import the checkpoints and test-dataset from the folder titled `motiontaskfiles` from this [folder](https://drive.google.com/drive/folders/1MgfStqB3Nx0tfnJRXc6n0vvoWoEt6N8H?usp=sharing). The video outputs for optical flow are also present in the same link.


## Disparity Task

We experimented the paper "Superpixel Segmentation using fully connected convolutional networks" and thier application of using super pixel network for upsampling stereo images. We achieved very good results compared to the baseline. You can find the results of the baseline [here](https://drive.google.com/drive/folders/1MgfStqB3Nx0tfnJRXc6n0vvoWoEt6N8H?usp=sharing) Import the checkpoints and test-dataset from the folder titled `disptaskfiles` in the same link.


Future Works
[TODO]  : Follow the same routine, but train the U-Net for [multi-object video segmentation using Davis Dataset](https://davischallenge.org/davis2017/code.html#unsupervised)
[TODO]  : Motion Segmentation Pipeline a bayesian learning framework instead of relying on Residual UNets.
