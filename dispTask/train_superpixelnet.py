
import torchvision.transforms as transforms
from dataset.sceneflowdataloader import SceneFlowLoader
import torch

from glob import glob
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import argparse
import shutil
import time
from tqdm import tqdm

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import superpixelnet.flow_transforms as flow_transforms
from superpixelnet.models.Spixel_single_layer import SpixelNet1l_bn
from superpixelnet.loss import compute_semantic_pos_loss
import datetime
from tensorboardX import SummaryWriter

from superpixelnet.train_util import *

# psmnet
from  models import *
from models.submodule import disparityregression


best_EPE = -1
n_iter = 0

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

class ARG:
    def __init__(self):
        self.dataset = 'SceneFlow'
        self.arch = 'SpixelNet1l_bn'
        # self.data= './data_preprocessing/Data'; self.data= './NYU'
        self.data = "./dataset/Monkaa"
        self.savepath = './checkpoints'
        self.train_img_height = 256; self.train_img_width= 512 
        # self.train_img_height = 128; self.train_img_width= 256
        self.input_img_height = 256;self.input_img_width = 512 
        
        self.workers = 4; self.epochs = 300  *10000
        self.start_epoch = 0; self.epoch_size = 6000; self.batch_size = 8;
        self.solver = 'adam'; self.lr= 0.00005; 
        self.momentum = 0.9; self.beta = 0.999; self.weight_decay=4e-4;self.bias_decay=0
        self.milestones=[200000]; self.additional_step=100000; 
        self.pos_weight = 0.003; self.downsize = 16;
        self.gpu = '0'; self.print_freq = 10; self.record_freq  = 5; self.label_factor=5; self.pretrained = "./checkpoints/SceneFlow/68epochs/model_best.tar";
        self.no_date=True
        
        self.maxdisp = 192; self.psmmodel = 'basic'
        self.seed = 1; self.pretrainedpsmnet = None

args = ARG()

# !----- NOTE the current code does not support cpu training -----!
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print('Current code does not support CPU training! Sorry about that.')
    exit(1)

N_CLASSES = 50

def main():
    global args, best_EPE, save_path, intrinsic, N_CLASSES

    # ============= savor setting ===================
    save_path = '{}_{}_{}epochs{}_b{}_lr{}_posW{}'.format(
        args.arch,
        args.solver,
        args.epochs,
        '_epochSize'+str(args.epoch_size) if args.epoch_size > 0 else '',
        args.batch_size,
        args.lr,
        args.pos_weight,
    )
    if not args.no_date:
        timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M")
    else:
        timestamp = ''
    save_path = os.path.abspath(args.savepath) + '/' + os.path.join(args.dataset, save_path  +  '_' + timestamp )

    # ==========  Data loading code ==============
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
    ])

    val_input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
    ])

    target_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
    ])

    co_transform = flow_transforms.Compose([
            flow_transforms.RandomCrop((args.train_img_height ,args.train_img_width))
        ])

    trainset = SceneFlowLoader(args.data, mode='train', transform=input_transform, target_transform=target_transform, co_transform=co_transform)
    valset = SceneFlowLoader(args.data, mode='val', transform=val_input_transform, target_transform=target_transform, co_transform=co_transform)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=False, drop_last=True)


    # import models.stackhourglass.PSMNet as stackhourglass
    torch.manual_seed(args.seed)
    if device.type  == 'cuda':
        torch.cuda.manual_seed(args.seed)


    # ============== create model ====================
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        args.arch = network_data['arch']
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        network_data = None
        print("=> creating model '{}'".format(args.arch))

    model = SpixelNet1l_bn( data = network_data).cuda()
    model = torch.nn.DataParallel(model).cuda()


    # if args.psmmodel == 'stackhourglass':
    #     psmnet = stackhourglass(args.maxdisp, slicmode=False)
    # elif args.psmmodel == 'basic':
    #     psmnet = basic(args.maxdisp)
    # else:
    #     print('no model')
    # psmnet = torch.nn.DataParallel(psmnet)
    # psmnet.to(device)

    # if args.pretrainedpsmnet is not None:
    #     print('Load pretrained model')
    #     pretrain_dict = torch.load(args.pretrainedpsmnet)
    #     psmnet.load_state_dict(pretrain_dict['state_dict'])

    # print('Number of PSM model parameters: {}'.format(sum([p.data.nelement() for p in psmnet.parameters()])))


    cudnn.benchmark = True
    #=========== creat optimizer, we use adam by default ==================
    assert(args.solver in ['adam', 'sgd'])
    print('=> setting {} solver'.format(args.solver))

    param_groups = [{'params': model.module.bias_parameters(), 'weight_decay': args.bias_decay},
                    {'params': model.module.weight_parameters(), 'weight_decay': args.weight_decay},
                    ]
    # param_groups.extend([{'params': psmnet.module.bias_parameters(), 'weight_decay': args.weight_decay},
    #                     {'params': psmnet.module.weight_parameters(), 'weight_decay': args.weight_decay}
    #                     ])
    if args.solver == 'adam':
        optimizer = torch.optim.Adam(param_groups, args.lr,
                                        betas=(args.momentum, args.beta))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(param_groups, args.lr,
                                    momentum=args.momentum)

    # for continues training
    if args.pretrained and ('dataset' in network_data):
        if args.pretrained and args.dataset == network_data['dataset'] :
            optimizer.load_state_dict(network_data['optimizer'])
            best_EPE = network_data['best_EPE']
            args.start_epoch = network_data['epoch']
            save_path = os.path.dirname(args.pretrained)



    print('=> will save everything to {}'.format(save_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_writer = SummaryWriter(os.path.join(save_path, 'train'))
    val_writer = SummaryWriter(os.path.join(save_path, 'val'))

    # spixelID: superpixel ID for visualization,
    # XY_feat: the coordinate feature for position loss term
    spixelID, XY_feat_stack = init_spixel_grid(args)
    val_spixelID,  val_XY_feat_stack = init_spixel_grid(args, b_train=False)

    #===================================================================================
    #=============================== Train routine =====================================
    #===================================================================================

    for epoch in range(args.start_epoch, args.epochs): # epochs chosen to be 3e6 because we want to end training based on iterations - 300k
        # train for one epoch
        train_avg_slic, train_avg_sem, iteration = train(train_loader, (model, None), optimizer, epoch,
                                                            train_writer, spixelID, XY_feat_stack )
        if epoch % args.record_freq == 0:
            train_writer.add_scalar('Mean avg_slic', train_avg_slic, epoch)

        # evaluate on validation set and save the module( and choose the best)
        with torch.no_grad():
            avg_slic, avg_sem  = validate(val_loader, (model, None), epoch, val_writer, val_spixelID, val_XY_feat_stack)
            if epoch % args.record_freq == 0:
                val_writer.add_scalar('Mean avg_slic', avg_slic, epoch)

        rec_dict = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.module.state_dict(),
                'best_EPE': best_EPE,
                'optimizer': optimizer.state_dict(),
                'dataset': args.dataset
            }

        if (iteration) >= (args.milestones[-1] + args.additional_step):
            save_checkpoint(rec_dict, is_best =False, filename='%d_step.tar' % iteration)
            print("Train finished!")
            break

        if best_EPE < 0:
            best_EPE = avg_sem
        is_best = avg_sem < best_EPE
        best_EPE = min(avg_sem, best_EPE)
        save_checkpoint(rec_dict, is_best)

def computeOutput(cost, Ql, S):
    upsampled_cost = upfeat(cost, Ql, S, S)
    assert upsampled_cost.size()[2] == Ql.size()[2]
    assert upsampled_cost.size()[3] == Ql.size()[3]
    upsampled_cost = F.upsample(upsampled_cost.unsqueeze(1), [args.maxdisp,Ql.size()[2],Ql.size()[3]], mode='trilinear').squeeze(1)
    output = F.softmax(upsampled_cost)
    output = disparityregression(args.maxdisp)(output)
    return output

def train(train_loader, networks, optimizer, epoch, train_writer, init_spixl_map_idx, XY_feat_stack):
    global n_iter, args, intrinsic, N_CLASSES

    batch_time = AverageMeter()
    data_time = AverageMeter()

    total_loss = AverageMeter()
    losses_slic = AverageMeter()
    losses_psm = AverageMeter()

    epoch_size =  len(train_loader) if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)

    model, psmnet = networks
    # switch to train mode
    model.train()
    # psmnet.train()
    end = time.time()
    iteration = 0

    for i, sample in enumerate(train_loader):
        imL, imR, label, labelR, disp_true = to_device(sample, device)
        iteration = i + epoch * epoch_size

        # ========== adjust lr if necessary  ===============
        if (iteration + 1) in args.milestones:
            state_dict = optimizer.state_dict()
            for param_group in state_dict['param_groups']:
                param_group['lr'] = args.lr * ((0.5) ** (args.milestones.index(iteration + 1) + 1))
            optimizer.load_state_dict(state_dict)

        # ========== complete data loading ================
        label_1hot = label2one_hot_torch(label, C=N_CLASSES) # set C=50 as SSN does
        LABXY_feat_tensor = build_LABXY_feat(label_1hot, XY_feat_stack)  # B* (50+2 )* H * W

        torch.cuda.synchronize()
        data_time.update(time.time() - end)

        # ========== predict association map ============
        Ql = model(imL)
        # ========== compute slic loss ============
        loss, psmloss, slic_loss = compute_semantic_pos_loss( Ql, LABXY_feat_tensor,
                                                                pos_weight= args.pos_weight, kernel_size=args.downsize)

        # NOTE: all references of loss_sem will be psmloss and loss_pos will be slic_loss

        # ========= back propagate ===============

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ========  measure batch time ===========
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # =========== record and display the loss ===========
        # record loss and EPE
        total_loss.update(loss.item(), imL.size(0))
        losses_slic.update(slic_loss.item(), imL.size(0))
        losses_psm.update(psmloss.item(), imL.size(0))


        if i % args.print_freq == 0:
            print('train Epoch: [{0}][{1}/{2}]\t Time {3}\t Data {4}\t Total_loss {5}\t Loss_sem {6}\t Loss_pos {7}\t'
                    .format(epoch, i, epoch_size, batch_time, data_time, total_loss, losses_slic, losses_psm))

            train_writer.add_scalar('Train_loss', slic_loss.item(), i + epoch*epoch_size)
            train_writer.add_scalar('learning rate',optimizer.param_groups[0]['lr'], i + epoch * epoch_size)

        n_iter += 1
        if i >= epoch_size:
            break

        if (iteration) >= (args.milestones[-1] + args.additional_step):
            break
        
    # =========== write information to tensorboard ===========
    if epoch % args.record_freq == 0:
        train_writer.add_scalar('Train_loss_epoch', loss.item(),  epoch )
        train_writer.add_scalar('loss_sem',  psmloss.item(),  epoch )
        train_writer.add_scalar('loss_pos',  slic_loss.item(), epoch)

        #save image
        mean_values = torch.tensor([0.411, 0.432, 0.45], dtype=imL.dtype).view(3, 1, 1)
        input_l_save = (make_grid((imL.detach().cpu() + mean_values).clamp(0, 1), nrow=args.batch_size))
        label_save = make_grid(args.label_factor * label)

        train_writer.add_image('Input', input_l_save, epoch)
        train_writer.add_image('label', label_save, epoch)

        # init_spixl_map_idx = spixelID
        curr_spixl_map = update_spixl_map(init_spixl_map_idx,Ql)
        spixel_lab_save = make_grid(curr_spixl_map, nrow=args.batch_size)[0, :, :]
        spixel_viz, _ = get_spixel_image(input_l_save, spixel_lab_save)
        train_writer.add_image('Spixel viz', spixel_viz, epoch)

        # save associ map,  --- for debug only
        _, prob_idx = torch.max(Ql, dim=1, keepdim=True)
        prob_map_save = make_grid(assign2uint8(prob_idx))
        train_writer.add_image('assigment idx', prob_map_save, epoch)

        print('==> write train step %dth to tensorboard' % i)

    return total_loss.avg, losses_psm.avg, iteration

def validate(val_loader, networks, epoch, val_writer, init_spixl_map_idx, XY_feat_stack):
    global n_iter,   args,    intrinsic
    batch_time = AverageMeter()
    data_time = AverageMeter()

    total_loss = AverageMeter()
    losses_psm = AverageMeter()
    losses_slic = AverageMeter()

    # set the validation epoch-size, we only randomly val. 400 batches during training to save time
    epoch_size = min(len(val_loader), 400)

    model, psmnet = networks
    # switch to eval mode
    model.eval()
    # psmnet.eval()
    end = time.time()

    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            imL, imR, label, labelR, disp_true = to_device(sample, device)

            # ========== complete data loading ================
            label_1hot = label2one_hot_torch(label, C=N_CLASSES) # set C=50 as SSN does
            LABXY_feat_tensor = build_LABXY_feat(label_1hot, XY_feat_stack)  # B* (50+2 )* H * W

            torch.cuda.synchronize()
            data_time.update(time.time() - end)


            # ========== predict association map ============
            Ql = model(imL)
            # Qr = model(imR)
            # ========== compute slic loss ============
            loss, psmloss, slic_loss = compute_semantic_pos_loss( Ql, LABXY_feat_tensor,
                                                                    pos_weight= args.pos_weight, kernel_size=args.downsize)

            # NOTE: all references of loss_sem will be psmloss and loss_pos will be slic_loss

            # measure elapsed time
            torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            end = time.time()

            # record loss and EPE
            total_loss.update(loss.item(), imL.size(0))
            losses_psm.update(psmloss.item(), imL.size(0))
            losses_slic.update(slic_loss.item(), imL.size(0))

            if i % args.print_freq == 0:
                print('val Epoch: [{0}][{1}/{2}]\t Time {3}\t Data {4}\t Total_loss {5}\t Loss_sem {6}\t Loss_pos {7}\t'
                    .format(epoch, i, epoch_size, batch_time, data_time, total_loss, losses_psm, losses_slic))

            if i >= epoch_size:
                break

    # =============  write result to tensorboard ======================
    if epoch % args.record_freq == 0:
        val_writer.add_scalar('val_loss_epoch', loss.item(), epoch)
        val_writer.add_scalar('val_loss_sem', psmloss.item(), epoch)
        val_writer.add_scalar('val_loss_pos', slic_loss.item(), epoch)

        mean_values = torch.tensor([0.411, 0.432, 0.45], dtype=imL.dtype).view(3, 1, 1)
        input_l_save = (make_grid((imL.detach().cpu() + mean_values).clamp(0, 1), nrow=args.batch_size))


        curr_spixl_map = update_spixl_map(init_spixl_map_idx, Ql)
        spixel_lab_save = make_grid(curr_spixl_map, nrow=args.batch_size)[0, :, :]
        spixel_viz, _ = get_spixel_image(input_l_save, spixel_lab_save)

        label_save = make_grid(args.label_factor * label)

        val_writer.add_image('Input', input_l_save, epoch)
        val_writer.add_image('label', label_save, epoch)
        val_writer.add_image('Spixel viz', spixel_viz, epoch)
        
        # --- for debug
        #     _, prob_idx = torch.max(assign, dim=1, keepdim=True)
        #     prob_map_save = make_grid(assign2uint8(prob_idx))
        #     val_writer.add_image('assigment idx level %d' % j, prob_map_save, epoch)

        print('==> write val step %dth to tensorboard' % i)

    return total_loss.avg, losses_psm.avg


def save_checkpoint(state, is_best, filename='checkpoint.tar'):
    torch.save(state, os.path.join(save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.tar'))



if __name__ == '__main__':
    main()
