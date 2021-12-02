import cv2, glob, torch, os, time
import numpy as np
import utils.normal_flow as nf
import utils.airsim_utils as au
from scipy.spatial.transform import Rotation as R
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

def tartanIntrinsics(fx = 320, fy=320, cx = 320.0, cy = 240.0):
    # fx = 320.0; fy = 320.0; cx = 320.0; cy = 240.0;
    # fov = 90; width = 640; height = 480
    K = np.eye(3)
    K[0,0] = fx; K[1,1] = fy; K[0,2] = cx; K[1,2]=cy
    return K
def TrainValSplit(master_samples, ratio = 0.1, seed = 42):
    dummy_Y = range(len(master_samples))
    train_samples, val_samples, _, _ = train_test_split(master_samples, dummy_Y, test_size=ratio, random_state=seed)
    return train_samples, val_samples
def gray(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
def cropcenter(img, shape):
    th,tw = shape
    w,h = img.shape[:2]
    x1 = int((w-tw)/2)
    y1 = int((h-th)/2)
    if len(img.shape)==3:
        return img[y1:y1+th,x1:x1+tw,:]
    else:
        return img[y1:y1+th,x1:x1+tw]
def crop64(img):    
    h,w = img.shape[:2]
    th, tw = (h//64)*64, (w//64)*64 
    x1 = int((w-tw)/2); y1 = int((h-th)/2)
    if len(img.shape)==3:
        return img[y1:y1+th,x1:x1+tw,:]
    else:
        return img[y1:y1+th,x1:x1+tw]


def computeGTFOE(tr, intrinsics):
    foe = (1/tr[2]) * np.array(( tr[0],  tr[1], tr[2])) # passing this gets closer estimates
    # foe[0]*=intrinsics[0,0]; foe[1]*=intrinsics[1,1]
    tr_gt = torch.from_numpy(foe[:2])
    return tr_gt


class Downscale(object):
    """
    Scale the flow and mask to a fixed size
    """
    def __init__(self, scale=4):
        '''
        size: output frame size, this should be NO LARGER than the input frame size! 
        '''
        self.downscale = 1.0/scale

    def __call__(self, sample): 
        if self.downscale !=1.0:
            if len(sample.shape)==3:
                # flow : 
                if sample.shape[2] == 2:
                        flow = cv2.resize(sample, (0, 0), fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_LINEAR)
                        flow[:, :, 0] = flow[:, :, 0] * self.downscale
                        flow[:, :, 1] = flow[:, :, 1] * self.downscale
                        return flow       
                # image:
                elif sample.shape[2] == 3:
                    return cv2.resize(sample, (0, 0), fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_LINEAR)                         
            else:                
                # intrinsics: 
                if sample.shape[0] == 3 :
                    sample[0] *=  self.downscale
                    sample[1] *=  self.downscale            
                    return sample
                # depth
                else:
                    return cv2.resize(sample, (0, 0), fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_LINEAR)
        else:
            return sample

def compute_K_layer(K, shape):
    def scaleIntrinsics(sample, downscalefactor = 0.25):
        sample[0] *=  downscalefactor
        sample[1] *=  downscalefactor             
        return sample
    fx,fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    sf =0.25
    shape = shape*sf
    h,w = shape
    K = scaleIntrinsics(K, sf)
    ww, hh = np.meshgrid(range(w), range(h))
    ww = (ww.astype(np.float32) - cx + 0.5 )/fx
    hh = (hh.astype(np.float32) - cy + 0.5 )/fy
    intrinsicLayer = np.stack((ww,hh)).transpose(1,2,0)

    return intrinsicLayer


class TartanLoader(Dataset):
    def __init__(self, basepath, mode = 'train', scale = 2,  ttype = 'foe'):
        self.transform=None
        self.basepath = basepath
        self.downsize = Downscale(scale)
        self.downsize4 = Downscale(4)
        
        self.samples = []
        self.ttype = ttype
        self.mode = mode
        for path in basepath:
            # LOAD ALL SAMPLES
            self.loadfiles(path)
        self.intrinsics = tartanIntrinsics()

        if mode == 'train':            
        #     self.samples, _ = TrainValSplit(self.samples, ratio = 0.1)
            print("Total samples = ", len(self.samples))
        # elif mode == 'val':
        #     _, self.samples = TrainValSplit(self.samples, ratio = 0.1 )
        #     print("Total samples = ", len(self.samples))
        else:
            print("Test samples = ", len(self.samples))
        # self.samples = self.samples[:100]        

    def loadfiles(self, path):

        folders = sorted(os.listdir(path+'/'))
        if self.mode != 'train' and self.mode != 'val':
            index = int(self.mode[-2:])
            iterator = folders[index:index+1]
        else:
            iterator = folders[:30]

        for folder in iterator:
            files = sorted(glob.glob(path+'/'+folder+'/image_left/*png'))
            flowfiles = sorted(glob.glob(path+'/'+folder+'/flow/*flow.npy'))
            flowfiles = sorted(glob.glob(path+'/'+folder+'/depth_left/*depth.npy'))

            poselist = np.loadtxt(path+'/'+folder+'/pose_left.txt').astype(np.float32)

            assert len(poselist) == len(files) ==len(flowfiles)+1, f"{len(poselist)} == {len(files)} == {len(flowfiles)+1}"
            assert(poselist.shape[1]==7) # position + quaternion
            
            print('Found {} image files in {}'.format(len(files), folder))
            n=1
            for i in range(len(files) - n):
                rpose = au.relative_pose(poselist[i], poselist[i+n], True, 'xyz') # n2c corrected
                # if rpose[2] > 0:
                # if (abs(rpose[0]/rpose[2])< 2 and abs(rpose[1]/rpose[2])< 2) :
                res = {'im0': files[i], 'im1': files[i+n], 'flow': flowfiles[i],'motion': rpose}                
                self.samples.append(res)
            
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # print(idx, self.samples[idx]['im0'])
        # get images
        im0 = cv2.imread(self.samples[idx]['im0'])
        im1 = cv2.imread(self.samples[idx]['im1'])
        depth = np.load(self.samples[idx]['flow'])
        # get relative pose, intrinsics and flow        
        rpose = self.samples[idx]['motion']
        intrinsics = self.intrinsics.copy()
        flow = np.load(self.samples[idx]['flow'])


        # crop to multiple of 64
        im0 = crop64(im0); im1 = crop64(im1); flow = crop64(flow); depth = crop64(depth)
        intrinsics[0,2]=im0.shape[1]//2; intrinsics[1,2]=im0.shape[0]//2 
        # resize if needed
        im0 = self.downsize(im0); im1 = self.downsize(im1)
        flow = self.downsize(flow); intrinsics = self.downsize(intrinsics)
        depth = self.downsize(depth)
        
        # flow = cv2.resize(self.downsize4(flow), ah(0, 0), fx=4, fy=4, interpolation=cv2.INTER_LINEAR) # downsize and then upsize without scaling the flow        
        im0, im1  = im0.transpose(2,0,1), im1.transpose(2,0,1)

        # this format fits tartan eval script
        rot_gt = np.array([rpose[3], rpose[4], rpose[5]])  # xyz
        tr_gt = np.array(( rpose[2],  rpose[0], rpose[1])) # zxy

        # convert and pass as output
        flow = flow.transpose(2,0,1)
        # nflow = torch.from_numpy(flow) # uncomment if you wanna pass optical flow
        return im0, im1, flow, depth, tr_gt, rot_gt, intrinsics 

