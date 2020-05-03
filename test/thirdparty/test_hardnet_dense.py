#!/usr/bin/python3 -utt
# -*- coding: utf-8 -*-
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import time
import os
#sys.path.insert(0, '/home/ubuntu/dev/opencv-3.1/build/lib')
import cv2
import math
import numpy as np
from PIL import Image


hardnet_base_path='../../thirdparty/hardnet/'


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1, keepdim = True) + self.eps)
        x= x / norm.expand_as(x)
        return x

class LocalNorm2d(nn.Module):
    def __init__(self, kernel_size = 32):
        super(LocalNorm2d, self).__init__()
        self.ks = kernel_size
        self.pool = nn.AvgPool2d(kernel_size = self.ks, stride = 1,  padding = 0)
        self.eps = 1e-10
        return
    def forward(self,x):
        pd = int(self.ks/2)
        mean = self.pool(F.pad(x, (pd,pd,pd,pd), 'reflect'))
        return torch.clamp((x - mean) / (torch.sqrt(torch.abs(self.pool(F.pad(x*x,  (pd,pd,pd,pd), 'reflect')) - mean*mean )) + self.eps), min = -6.0, max = 6.0)
    
class DenseHardNet(nn.Module):
    """HardNet model definition
    """
    def __init__(self, _stride = 2):
        super(DenseHardNet, self).__init__()
        self.input_norm = LocalNorm2d(17)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=_stride, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=_stride,padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, kernel_size=8, bias = False),
            nn.BatchNorm2d(128, affine=False),
            L2Norm()
        )
        return

    def forward(self, input, upscale = False):
        if input.size(1) > 1:
            feats = self.features(self.input_norm(input.mean(dim = 1, keepdim = True)))
        else:
            feats = self.features(self.input_norm(input))
        if upscale:
            return F.upsample(feats, (input.size(2), input.size(3)),mode='bilinear')
        return feats
    
def load_grayscale_var(fname):
    img = Image.open(fname).convert('RGB')
    img = np.mean(np.array(img), axis = 2) # convert to gray scale by averaging R-G-B channels
    var_image = torch.autograd.Variable(torch.from_numpy(img.astype(np.float32)), volatile = True)
    var_image_reshape = var_image.view(1, 1, var_image.size(0),var_image.size(1))
    return var_image_reshape

if __name__ == '__main__':
    DO_CUDA = True
    UPSCALE = False
    stride = 2;
    try:
        if len(sys.argv) > 1:         
            input_img_fname = sys.argv[1]
            output_fname = sys.argv[2]
            if len(sys.argv) > 3:
                DO_CUDA = sys.argv[3] != 'cpu'
            if len(sys.argv) > 4:
                UPSCALE = sys.argv[4] == 'UPSCALE'
                if  sys.argv[4] == 'NOSTRIDE':
                    stride = 1
        else:
            input_img_fname = '../data/kitti06-12-color.png'
            output_fname = 'hardnet.out.txt'
            DO_CUDA = True             
    except:
          #print("Wrong input format. Try ./extract_DenseHardNet.py imgs/ref.png out.txt gpu")
          print("Wrong input format. Try " + sys.argv[0] + " imgs/ref.png out.txt gpu")          
          sys.exit(1)
    model_weights = hardnet_base_path + 'pretrained/pretrained_all_datasets/HardNet++.pth'
    model = DenseHardNet(stride)
    checkpoint = torch.load(model_weights)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    img = load_grayscale_var(input_img_fname)
    if DO_CUDA:
        model.cuda()
        img = img.cuda()
        print('Extracting on GPU')
    else:
        print('Extracting on CPU')
        model = model.cpu()
    t = time.time()
    with torch.no_grad():
        desc = model(img, UPSCALE)
    et  = time.time() - t
    print('processing', et)
    desc_numpy = desc.cpu().detach().float().squeeze().numpy();
    desc_numpy = np.clip(((desc_numpy + 0.45) * 210.0).astype(np.int32), 0, 255).astype(np.uint8)
    print(desc_numpy.shape)
    np.save(output_fname, desc_numpy)