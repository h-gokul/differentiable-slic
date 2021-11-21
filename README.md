# differentiable-slic

The data preparation is same as https://github.com/NVlabs/ssn_superpixels.git.
To enforce connectivity in superpixels, the cython script takes from official code.
To simplify the implementation, each init superpixel has the same number of pixels during the training.


# References
- Check out  [SSN network](https://github.com/CYang0515/pytorch_ssn) to find the base repository and the pretrained model file
- Check [Tartan VO](https://github.com/castacks/tartanvo/blob/main/Datasets/tartanTrajFlowDataset.py) for the TartanOdometry code
- Check [TartanAir Dataset](https://github.com/castacks/tartanair_tools) for the dataset  download links
