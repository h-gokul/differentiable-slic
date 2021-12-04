import cv2, glob, os, time
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from pytorch_ssn.dataset import Resize, ssn_preprocess
from skimage.color import rgb2lab
from skimage.util import img_as_float
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
                # depth:
                elif sample.shape[2] == 1:
                    return cv2.resize(sample, (0, 0), fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_NEAREST)
            else:                
                # intrinsics: 
                if sample.shape[0] == 3 :
                    sample[0] *=  self.downscale
                    sample[1] *=  self.downscale            
                    return sample                    
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

def ned2cam(axis):
    return np.array((axis[1], axis[2], axis[0]))

def relative_orientation(q_start, q_end, mode):
    r_start = R.from_quat(q_start)
    r_end   = R.from_quat(q_end)
    r_diff = r_start.inv() * r_end
    return r_diff.as_euler(mode)

def relative_pose(apose0, apose1, n2c=True, mode = 'xyz'):
    # last_pos, last_q are reference frames
    last_pos = apose0[:3]; pos = apose1[:3]
    last_q = apose0[3:]; q = apose1[3:]

    #### obtain relative translation (dp_cam)
    dp = pos - last_pos 
    dp_body = R.from_quat(last_q).inv().as_matrix() @ dp  # R_inv dt
    if n2c:
        r_p = ned2cam(dp_body) # convert from ned to camera frame
    else:
        r_p = dp_body
    #### obtain relative orientation (body_rates_ypr)- yaw pitch roll
    r_q = relative_orientation(last_q, q, mode)   
    
    return np.concatenate((r_p, r_q), axis=0)

def u_rot(roll, pitch, yaw, intrinsics, grid_size):
# def u_rot(yaw, pitch, roll, intrinsics, grid_size):

    a = pitch
    b = yaw
    g = roll
    f = intrinsics[0,0]
    H,W = grid_size
    cx, cy = intrinsics[0,2], intrinsics[1,2]

    (x, y) = np.meshgrid(np.arange(0, W, dtype=np.float32) - cx + 0.5,
                         np.arange(0, H, dtype=np.float32) - cy + 0.5)

    # Equations 1 and 2 from page 3 of Passive Navigation as a Pattern Recognition Problem
    u_rot = a*x*y/f - b*(x*x/f + f) + g*y
    v_rot = a*(y*y/f+f) - b*x*y/f - g*x

    return np.dstack((u_rot, v_rot))

def u_trans(v_x, v_y, v_z, depth, intrinsics):
    u = v_x
    v = v_y
    w = v_z
    Z = depth
    f = intrinsics[0,0]
    cx, cy = intrinsics[0,2], intrinsics[1,2]
    H,W = depth.shape
        
    (x, y) = np.meshgrid(np.arange(0, W, dtype=np.float32) - cx + 0.5,
                         np.arange(0, H, dtype=np.float32) - cy + 0.5)

    # Equations 1 and 2 from page 3 of Passive Navigation as a Pattern Recognition Problem
    # Substitute x_0 and y_0
    u_t = (w * x - u * f) / Z
    v_t = (w * y - v * f) / Z

    return np.dstack((u_t, v_t))

def compute_optical_flow(rpose, intrinsics, depth):
    flow_rot = u_rot(rpose[3], rpose[4], rpose[5], intrinsics, grid_size = depth.shape)
    flow_t = u_trans(rpose[0], rpose[1], rpose[2], depth,intrinsics)
    return flow_t + flow_rot

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
            print("Total samples = ", len(self.samples))
        else:
            print("Test samples = ", len(self.samples))

    def loadfiles(self, path):

        folders = sorted(os.listdir(path+'/'))
        if self.mode != 'train' and self.mode != 'val':
            index = int(self.mode[-2:])
            iterator = folders[index:index+1]
        else:
            iterator = folders[:6]

        for folder in iterator:
            files = sorted(glob.glob(path+'/'+folder+'/image_left/*png'))
            depthfiles = sorted(glob.glob(path+'/'+folder+'/depth_left/*depth.npy'))
            poselist = np.loadtxt(path+'/'+folder+'/pose_left.txt').astype(np.float32)
            assert len(poselist) == len(files) ==len(depthfiles), f"{len(poselist)} == {len(files)} == {len(depthfiles)+1}"
            assert(poselist.shape[1]==7) # position + quaternion
            
            print('Found {} image files in {}'.format(len(files), folder))
            n=1
            for i in range(len(files) - n):
                rpose = relative_pose(poselist[i], poselist[i+n], True, 'xyz') # n2c corrected
                # if rpose[2] > 0:
                # if (abs(rpose[0]/rpose[2])< 2 and abs(rpose[1]/rpose[2])< 2) :
                res = {'im0': files[i], 'im1': files[i+n], 'motion': rpose, 'depth': depthfiles[i]}                
                self.samples.append(res)
            
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # print(idx, self.samples[idx]['im0'])
        # get images
        im0 = cv2.imread(self.samples[idx]['im0'])
        im1 = cv2.imread(self.samples[idx]['im1'])

        # get relative pose, intrinsics and flow        
        rpose = self.samples[idx]['motion']
        intrinsics = self.intrinsics.copy()

        depth = np.load(self.samples[idx]['depth'])[..., np.newaxis]

        
        # crop to multiple of 64
        im0 = crop64(im0); im1 = crop64(im1); depth = crop64(depth)
        intrinsics[0,2]=im0.shape[1]//2; intrinsics[1,2]=im0.shape[0]//2 
        # resize if needed
        im0 = self.downsize(im0); im1 = self.downsize(im1)
        intrinsics = self.downsize(intrinsics)
        depth = self.downsize(depth)[..., np.newaxis]
        flow = compute_optical_flow(rpose, intrinsics, depth[...,0])

        h,w = im0.shape[:2]
        k = int(0.5 * (h*w)//25 )
        # print(k)
        ssn_inputs, ssn_args = ssn_preprocess(rgb2lab(im1.astype(np.float32)), None, k )


        # this format fits tartan eval script
        # rot_gt = np.array([rpose[3], rpose[4], rpose[5]])  # xyz
        rot_gt = np.array([rpose[4], rpose[5], rpose[3]])
        tr_gt = np.array(( rpose[0], rpose[1], rpose[2])) # zxy

        # convert and pass as output
        flow = flow.transpose(2,0,1).astype(np.float32)
        depth = depth.transpose(2,0,1).astype(np.float32)
        depth[depth>100]=0
        im0, im1  = im0.transpose(2,0,1).astype(np.float32), im1.transpose(2,0,1).astype(np.float32)
        flow_inliers = (np.abs(flow[0]) < 1000) & (np.abs(flow[1]) < 1000)

        # nflow = torch.from_numpy(flow) # uncomment if you wanna pass optical flow
        return im0, im1, flow, flow_inliers, depth, tr_gt, rot_gt, intrinsics, ssn_inputs, ssn_args

