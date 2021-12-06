import cv2
from torch.utils.data import Dataset, DataLoader
from glob import glob
from pytorch_ssn.dataset import Resize
from pytorch_ssn.model.SSN import SSN, crop_like
from pytorch_ssn.dataset import Resize, ssn_preprocess
from skimage.color import rgb2lab
from skimage.util import img_as_float
import os
import torch
import numpy as np
from tqdm import tqdm
from pytorch_ssn.IO import foldercheck
from pytorch_ssn.RAFT.core.raft import RAFT

os.environ['CUDA_VISIBLE_DEVICES']='0'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

# Helper Functions

def imgtensor2np(img):
    return img.permute(1,2,0).detach().cpu().numpy()
def to_device(args, device):
    args_out = []
    for arg in args:
        if isinstance(arg, list):
            arg = [ elem.to(device) for elem in arg ]
        else:
            arg = arg.to(device)
        args_out.append(arg)
    return args_out


class SLICGenLoader(Dataset):
    def __init__(self, basepath, folder, shape = (256,256)):
        self.imfiles = sorted(glob(f"{basepath}/JPEGImages/480p/{folder}/*jpg"))
        self.resize = Resize(shape)    
        print(f"Total samples: {len(self.imfiles)}")
        
    def __getitem__(self, idx):
        im1 = cv2.imread(self.imfiles[idx])
        im1 = self.resize(im1, im1.shape[:2])

        h,w = im1.shape[:2]
        k = int(0.5 * (h*w)//25 )
        ssn_inputs, ssn_args = ssn_preprocess(rgb2lab(img_as_float(im1)), None, k )
        return np.transpose(im1, [2, 0, 1]).astype(np.float32), ssn_inputs, ssn_args
    def __len__(self):
        return len(self.imfiles)


# [SETUP] - Raft model and SLIC
class MODELARGS:
    def __init__(self):
        self.ssn_dir = './pytorch_ssn/model/slic_model/45000_0.527_model.pt'
        self.model = "./pytorch_ssn/model/flow_model/raft-kitti.pth"    
        self.small= False; self.mixed_precision = True; self.alternate_corr=False; self.dropout = 0.0
        self.validate =False; self.add_noise=True
        self.clip = 1.0; self.gamma = 0.8; self.wdecay = .00005; self.epsilon=1e-8; self.iters=12
        self.batch_size = 6; self.epochs=20; self.lr = 0.00002


if __name__ == '__main__':
    args = MODELARGS()

    SSNLayer = SSN(args.ssn_dir, spixel_size=(5,5),dtype = 'layer', device = device)

    SSNLayer.eval()
    basepath='./Data/DAVIS'
    for folder in sorted(os.listdir(f"{basepath}/JPEGImages/480p")):
        print(f"Folder {folder}")
        slicdataset = SLICGenLoader(basepath, folder)
        data_loader = DataLoader(slicdataset, batch_size=1, shuffle=False, num_workers=4)
        foldercheck(f"{basepath}/SLICFeatures/480p/{folder}/")
        with torch.no_grad():
            for i, sample in tqdm(enumerate(data_loader)):
                im1,ssn_input = to_device(sample[:2], device)            
                ssn_params = to_device(sample[-1], device)
                ssn_params.extend([None])
                slic_features, spix_indices = SSNLayer(ssn_input, ssn_params) 
                slic_features = crop_like(slic_features, im1)
                savepath = f"{basepath}/SLICFeatures/480p/{folder}/{str(i).zfill(4)}.npy"
                np.save( savepath, imgtensor2np(slic_features[0]))

    print(" End SLIC generation")