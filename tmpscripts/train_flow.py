import os
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
# import cv2
# import matplotlib.pyplot as plt
# from pytorch_ssn.model.SSN import SSN, crop_like, superpixel_flow
# import pytorch_ssn.IO as IO
from pytorch_ssn.connectivity import enforce_connectivity
from pytorch_ssn.model.util import get_spixel_image
from pytorch_ssn.RAFT.core.raft import RAFT

os.environ['CUDA_VISIBLE_DEVICES']='0'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

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

from tartandataloader import TartanLoader 
from torch.utils.data import DataLoader
from loss import sequence_loss, EPE
from pytorch_ssn.IO import foldercheck
iters = 0

def Optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler


def Trainer(args, train_loader, net, optimizer, scaler, epoch):
    global iters, device
    losses = []
    net.train()
    iterator = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
    for i, sample in iterator:
        iters+=1
        im0, im1, flow, flow_inliers, _, _, _, _, = to_device(sample[:-2], device)
        
        if args.add_noise:
            stdv = np.random.uniform(0.0, 5.0)
            im0 = (im0 + stdv * torch.randn(*im0.shape).cuda()).clamp(0.0, 255.0)
            im1 = (im1 + stdv * torch.randn(*im1.shape).cuda()).clamp(0.0, 255.0)

        flow_preds = net(im0, im1, iters=args.iters)
        # epe.append(EPE(flow_pr, flow).item())

        loss, metrics = sequence_loss(flow_preds, flow, flow_inliers, args.gamma)

        iterator.set_description(f'Epoch [{epoch}/{args.epochs}]')
        iterator.set_postfix(loss=loss.item(), epe=metrics['epe'])

        # ssn_input = sample[-2].to(device)  
        # ssn_params = to_device(sample[-1], device)
        # ssn_params.extend([None])
        # _, spix_indices = SSNLayer(ssn_input, ssn_params) 
        # spix_indices = crop_like(spix_indices.unsqueeze(1), im1)
        # segflow_GT, _ = superpixel_flow( flow.clone(), spix_indices)
        # segflow_pred, _ = superpixel_flow( flow_preds[0], spix_indices)    
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)                
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)            
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()
        losses.append(loss.item())
    return np.mean(losses)

def validate(args, val_loader, net):
    global iters, device
    EPEs = []
    net.eval()
    iterator = tqdm(enumerate(val_loader), total=len(val_loader), leave=True)
    for i, sample in iterator:
        optimizer.zero_grad()
        im0, im1, flow, _, _, _, _, _, = to_device(sample[:-2], device)

        _, flow_pr = net(im0, im1, iters=args.iters, test_mode=True)
        epe = EPE(flow_pr, flow).item()
        EPEs.append(epe)
        iterator.set_postfix(epe=epe)
        # ssn_input = sample[-2].to(device)  
        # ssn_params = to_device(sample[-1], device)
        # ssn_params.extend([None])    
    return np.mean(EPEs)


# [SETUP] - Raft model and SLIC
class MODELARGS:
    def __init__(self):
        self.ssn_dir = './pytorch_ssn/model/slic_model/45000_0.527_model.pt'
        # self.model = "./pytorch_ssn/model/flow_model/raft-kitti.pth"    
        self.model = "./checkpoints/7_tartan.pth"
        self.small= False; self.mixed_precision = True; self.alternate_corr=False; self.dropout = 0.0
        self.validate =False; self.add_noise=True
        self.clip = 1.0; self.gamma = 0.8; self.wdecay = .00005; self.epsilon=1e-8; self.iters=12
        self.batch_size = 6; self.epochs=20; self.lr = 0.00002
args = MODELARGS()
valset = TartanLoader(basepath = ['Data/office2'], mode = 'test09', scale = 2)
val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=4)
print(f"Valset has {len(val_loader)} batches")
trainset = TartanLoader(basepath = ['Data/office2'], mode = 'train', scale = 2)
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=4)
print(f"Trainset has {len(train_loader)} batches")
args.num_steps = args.epochs * len(train_loader)

# slic layer
# SSNLayer = SSN(args.ssn_dir, spixel_size=(5,5),dtype = 'layer', device = device)

# flow network

net= RAFT(args)
print("Parameter Count: %d" % net.count_parameters())
net = torch.nn.DataParallel(net)
net.load_state_dict(torch.load(args.model))
print(f"Loading model from {args.model}")

net = net.to(device)

optimizer, scheduler = Optimizer(args, net)
scaler = GradScaler(enabled = args.mixed_precision)


foldercheck('checkpoints/')
best_error = float("inf")

for epoch in range(args.epochs):
    train_loss = Trainer(args, train_loader, net, optimizer, scaler, epoch)
    val_epe = validate(args, val_loader, net)

    print(f"**** END OF EPOCH: {epoch} || Train loss: {train_loss} || val epe: {val_epe}  **** ")

    if val_epe < best_error:
        PATH = f'checkpoints/{epoch}_tartan.pth' 
        print(f"savng model in {PATH}")
        torch.save(net.state_dict(), PATH)


