import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
def deconv(in_planes, out_planes, kernel_size=3, stride=2, padding=None, dilation=1, bn_layer=False, bias=True):
    # if padding is None:
    #     padding = (kernel_size-1)//2
    # nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, dilation, bias=False)
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding=padding, dilation=dilation, bias=bias),

        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.1,inplace=True)
    )


def conv(in_planes, out_planes, kernel_size=3, stride=2, padding=1, dilation=1, bn_layer=False, bias=True):
    if padding is None:
        padding = (kernel_size-1)//2

    if bn_layer:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
    else: 
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
            nn.ReLU(inplace=True)
        )
        
class ResBlock(nn.Module):
    expansion = 2
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(ResBlock, self).__init__()

        self.conv1 = conv(inplanes, planes, 3, stride, pad, dilation)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, pad, dilation)
        self.bn = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x); 
        out = self.conv2(out); 
        out = self.bn(out); 
        if self.downsample is not None:
            x = self.downsample(x)
        out += x 
        return F.relu(out, inplace=True)
        
class ResBlockTranspose(nn.Module):
    expansion = 0.5
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(ResBlockTranspose, self).__init__()
        self.conv1 = deconv(inplanes, planes, 3, stride, pad, dilation) 
        self.conv2 = nn.ConvTranspose2d(planes, planes, 3, 1, pad, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return F.relu(out, inplace=True)
def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]

class ResUnet(nn.Module):
    def __init__(self, args):
        super(ResUnet, self).__init__()
        inputnum = 7; outnum=1
        outputnums = np.array([16,32,64,128,256])
        if not args.small:
            outputnums*=2

        self.conv0 = conv(inputnum, outputnums[0], kernel_size=7, stride=2, padding=None, bn_layer=True)
        self.conv1 = conv(outputnums[0], outputnums[0], kernel_size=5, stride=2, padding=None, bn_layer=True)
        self.inplanes = outputnums[1]
        
        self.reslayer1 = self._down_res(ResBlock,outputnums[0], outputnums[1], blocks=1, stride=1, pad=1, dilation=1) 
        self.reslayer2 = self._down_res(ResBlock,outputnums[1], outputnums[2], blocks=1, stride=1, pad=1, dilation=1) 
        self.reslayer3 = self._down_res(ResBlock,outputnums[2], outputnums[3], blocks=1, stride=1, pad=1, dilation=1) 

        self.reslayer3T = self._up_res(ResBlockTranspose, outputnums[3],outputnums[2], blocks=1, stride=1, pad=1, dilation=1)         
        self.reslayer2T = self._up_res(ResBlockTranspose, outputnums[2],outputnums[1], blocks=1, stride=1, pad=1, dilation=1) 
        self.reslayer1T = self._up_res(ResBlockTranspose, outputnums[1],outputnums[0], blocks=1, stride=1, pad=1, dilation=1) 

        self.upconv1 = deconv(outputnums[0], outputnums[0], kernel_size=5, stride=2, padding=1, dilation=1, bn_layer=True)
        self.upconv0 = deconv(outputnums[0], outputnums[0], kernel_size=7, stride=2, padding=1, dilation=1, bn_layer=True)
        self.upconv_pred = deconv(outputnums[0], outnum, kernel_size=7, stride=1, padding=1, dilation=1, bn_layer=False)
        self.sigmoid = nn.Sigmoid()

    def _down_res(self, block, inplanes, planes, blocks, stride, pad, dilation):
        downsample = None
        downsample = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, pad, dilation))
        layers.append(nn.Conv2d(planes, planes, 3, 2, pad, dilation))
        return nn.Sequential(*layers)

    def _up_res(self, block,inplanes, planes, blocks, stride, pad, dilation):
        downsample = nn.ConvTranspose2d(inplanes, planes, kernel_size=1, stride=stride)
        layers = []
        layers.append(block(inplanes, planes, stride, downsample, pad, dilation))
        layers.append(nn.ConvTranspose2d(planes, planes, 3, 2, pad, dilation))
        return nn.Sequential(*layers)

    def correct_rgb(self, inputs, rgb_max = 255.0):
        
        x= inputs.contiguous() # no shape change
        x = x.view(inputs.size()[:2]+(-1,)) # b,c,seq_len*h*w
        x=x.mean(dim=-1) # get mean along b,c,seq_len*h*w
        rgb_mean=x.view(inputs.size()[:2] + (1,1,1)) # pad 1,1,1
        # or do this :
        # rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        x = (inputs - rgb_mean) / rgb_max
        return x
        
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def forward(self, x):
        # encoder
        down_conv0 = self.conv0(x)
        down_conv1 = self.conv1(down_conv0)
        down_resconv1 = self.reslayer1(down_conv1)
        down_resconv2 = self.reslayer2(down_resconv1)
        down_resconv3 = self.reslayer3(down_resconv2)

        # decoder
        up_resconv3 = self.reslayer3T(down_resconv3)
        up_resconv2 = self.reslayer2T(up_resconv3)
        up_resconv1 = self.reslayer1T(up_resconv2)
        up_conv1 = self.upconv1(up_resconv1) 
        up_conv1 = crop_like(up_conv1,down_conv0)
        up_conv0 = self.upconv0(up_conv1) 
        up_conv0 = crop_like(up_conv0,x)
        out = self.upconv_pred(up_conv0) 
        out = crop_like(out,x)

        out = self.sigmoid(out)
        return out



# def NFlowRes(data=None):
#     model = FlowRes()
#     if data is not None:
#         model.load_state_dict(data['state_dict'])
#     return model
