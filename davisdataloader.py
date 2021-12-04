import numpy as np
import cv2, torch, os
from torch.utils.data import Dataset
from skimage.color import rgb2lab
from skimage.util import img_as_float
from pytorch_ssn.dataset import Resize, ssn_preprocess
from glob import glob

class DAVISSegmentationLoader(Dataset):

    def __init__(self, basepath="./Data/DAVIS", mode = "val", shape = (256,256)):

        file= f"{basepath}/ImageSets/480p/{mode}.txt"
        with open(file) as f:
            lines = f.readlines()
        f.close()

        if mode == 'val':
            folders = sorted(os.listdir(f"{basepath}/JPEGImages/480p"))[40:]
        else:    
            folders = sorted(os.listdir(f"{basepath}/JPEGImages/480p"))[:40]
        self.samples = []
        for folder in folders:
            imfiles = sorted(glob(f"{basepath}/JPEGImages/480p/{folder}/*jpg"))
            gtfiles = sorted(glob(f"{basepath}/Annotations/480p/{folder}/*png"))
            assert len(gtfiles) == len(imfiles)
            for i in range(len(imfiles)-1):
                res = {'im1': imfiles[i], 'im2': imfiles[i+1], 'gt1': gtfiles[i], 'gt2': gtfiles[i+1] } 
                self.samples.append(res)
        self.samples= self.samples[:10]  
        self.resize = Resize(shape)
        
    def _loadBBoxes(self, mask):
		# Compute bounding boxes
        coords = np.where(mask!=0)
        if len(coords[0]) <=1:
            return None
        else:
            tl = np.min(coords[1]),np.min(coords[0])
            br = np.max(coords[1]),np.max(coords[0])
            return (tl[0],tl[1],br[0],br[1])

    def __getitem__(self, idx):
        im1, im2 = cv2.imread(self.samples[idx]['im1']),cv2.imread(self.samples[idx]['im2'])
        gt1, gt2 = cv2.imread(self.samples[idx]['gt1']),cv2.imread(self.samples[idx]['gt2'])
        im1, im2 = self.resize(im1, im1.shape[:2]), self.resize(im2, im2.shape[:2])
        gt1, gt2 = self.resize(gt1, gt1.shape[:2]), self.resize(gt2, gt2.shape[:2])

        h,w = im1.shape[:2]
        k = int(0.5 * (h*w)//25 )
        ssn_inputs, ssn_args = ssn_preprocess(rgb2lab(img_as_float(im1)), None, k )

        im1, im2 = np.transpose(im1, [2, 0, 1]).astype(np.float32), np.transpose(im2, [2, 0, 1]).astype(np.float32)
        gt1, gt2 = np.transpose(gt1, [2, 0, 1]).astype(np.float32), np.transpose(gt2, [2, 0, 1]).astype(np.float32)

        return im1, im2, gt1[:1]/255.0, gt2[:1]/255.0, ssn_inputs, ssn_args

    def __len__(self):
        return len(self.samples)

