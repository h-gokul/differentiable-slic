import torch.nn as nn
from .ssn_pytorch_joshuasmith44.model import SSNModel
import sys
import math
import torch
import matplotlib.pyplot as plt

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixerSSN(nn.Module):
    def __init__(self, dim=256, depth=8, kernel_size=9, patch_size=8, in_chans=3, num_classes=10, activation=nn.ReLU, pos_scale = 2.5, color_scale = 0.26 , **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.device = torch.device('cuda')
        self.num_features = dim
        self.pos_scale = pos_scale
        self.color_scale = color_scale
        self.patch_size = patch_size
        self.head = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()
        self.stem = nn.Sequential(
            nn.Conv2d(9 * in_chans, dim, kernel_size=patch_size, stride=patch_size),
            activation(),
            nn.BatchNorm2d(dim)
        )
        self.blocks = nn.Sequential(
            *[nn.Sequential(
                    Residual(nn.Sequential(
                        nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                        activation(),
                        nn.BatchNorm2d(dim)
                    )),
                    nn.Conv2d(dim, dim, kernel_size=1),
                    activation(),
                    nn.BatchNorm2d(dim)
            ) for i in range(depth)]
        )
        self.pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.nspix = (32 * 32) / (patch_size * patch_size)
        if (32 * 32) % (patch_size * patch_size) != 0:
            print(self.nspix)
            sys.exit("Patch size and cifar size are not compatible")
        self.SSNModel = SSNModel(10, nspix = self.nspix, n_iter = 10)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
          
    def forward_features(self, x):
        width = 32
        height = 32
        nspix_per_axis = int(math.sqrt(self.nspix))
        pos_scale = self.pos_scale * max(nspix_per_axis/height, nspix_per_axis/width)    

        coords = torch.stack(torch.meshgrid(torch.arange(height, device=self.device), torch.arange(width, device=self.device)), 0)
        coords = coords[None].repeat(x.shape[0], 1, 1, 1).float()

        inputs = torch.cat([self.color_scale*x, pos_scale*coords], 1)
        att, hard_labels, spix_feats, spixel_height, spixel_width = self.SSNModel(inputs)
        #print(spixel_height, spixel_width)
        #spixel_height = self.patch_size
        #spixel_width = self.patch_size

        att = torch.unsqueeze(att, 2)
        spixelPadder = nn.ZeroPad2d((spixel_width, spixel_width, spixel_height, spixel_height))
        x_pad = spixelPadder(x)
        br = x_pad[:, :, 2 * spixel_height: , 2 * spixel_width:] * att[0]
        bm = x_pad[:, :, 2 * spixel_height:, spixel_width : spixel_width + width] * att[1]
        bl = x_pad[:, :, 2 * spixel_height:, 0: width] * att[2]
        mr = x_pad[:, :, spixel_height:spixel_height + height , 2 * spixel_width:] * att[3]
        mm = x_pad[:, :, spixel_height:spixel_height + height, spixel_width : spixel_width + width] * att[4]
        ml = x_pad[:, :, spixel_height:spixel_height + height, 0: width] * att[5]
        tr = x_pad[:, :, 0:height, 2 * spixel_width:] * att[6]
        tm = x_pad[:, :, 0:height, spixel_width : spixel_width + width] * att[7]
        tl = x_pad[:, :, 0:height, 0: width] * att[8]
        un_attended_inp = torch.cat((br, bm, bl, mr, mm, ml, tr, tm, tl), dim = 1)
        x = self.stem(un_attended_inp)
        x = self.blocks(x)
        x = self.pooling(x)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
