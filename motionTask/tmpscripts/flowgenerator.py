import cv2
from torch.utils.data import Dataset, DataLoader
from glob import glob
from pytorch_ssn.dataset import Resize

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


class FlowGenLoader(Dataset):
    def __init__(self, basepath, folder, shape = (256,256)):
        imfiles = sorted(glob(f"{basepath}/JPEGImages/480p/{folder}/*jpg"))
        self.samples = []
        for i in range(len(imfiles)-1):
            res = {'im1': imfiles[i], 'im2': imfiles[i+1]} 
            self.samples.append(res)
        self.resize = Resize(shape)    
        print(f"Total samples: {len(self.samples)}")
        
    def __getitem__(self, idx):
        im1, im2 = cv2.imread(self.samples[idx]['im1']),cv2.imread(self.samples[idx]['im2'])
        im1, im2 = self.resize(im1, im1.shape[:2]), self.resize(im2, im2.shape[:2])
        return np.transpose(im1, [2, 0, 1]).astype(np.float32), np.transpose(im2, [2, 0, 1]).astype(np.float32)
    def __len__(self):
        return len(self.samples)


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

    net = torch.nn.DataParallel(RAFT(args))
    net.load_state_dict(torch.load(args.model))
    net = net.module.to(device)
    print("Parameter Count: %d" % net.count_parameters())

    net.eval()
    basepath='./Data/DAVIS'
    for folder in sorted(os.listdir(f"{basepath}/JPEGImages/480p")):
        print(f"Folder {folder}")
        flowdataset = FlowGenLoader(basepath, folder)
        data_loader = DataLoader(flowdataset, batch_size=1, shuffle=False, num_workers=4)
        foldercheck(f"{basepath}/OpticalFlow/480p/{folder}/")
        with torch.no_grad():
            for i, sample in tqdm(enumerate(data_loader)):
                im1, im2 = to_device(sample, device)            
                savepath = f"{basepath}/OpticalFlow/480p/{folder}/{str(i).zfill(4)}.npy"
                _, flow_pr = net(im1, im2, iters=args.iters, test_mode=True)        
                np.save( savepath, imgtensor2np(flow_pr[0]))

    print(" End flow generation")