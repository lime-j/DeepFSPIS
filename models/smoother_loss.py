import torch.nn as nn
import torch.nn.functional as F
import torch
#import pytorch_colors as colors

import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import math
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

#vgg = models.vgg19(pretrained=True).features
#print("vgg:  ",vgg)
#vgg = vgg.cuda()


#content_layers_default = ['conv_2']
#style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

class SmootherLoss(nn.Module):
    def __init__(self):
        super(SmootherLoss, self).__init__()
        self.criterionMSE = nn.MSELoss()
        self.criterionL1 = nn.SmoothL1Loss()
        self.criterionCLS = nn.BCELoss()
        self.a = 0.1
        self.u = 4
        self.b = 45 
        self.e = 0.005
        self.cnt = 0

    def forward(self, genimgs, targetimgs, masks , zero, lamb, stage):

        masks_3c = torch.cat([masks, masks, masks], dim=1)
        batch_size = genimgs.size()[0]

        channels = genimgs.size()[1] 
        h_img = genimgs.size()[2]
        w_img = genimgs.size()[3]

        # compute struct image gradient
        gradient_genimg_h = F.pad(torch.abs(genimgs[:, :, 1:, :] - genimgs[:, :, :h_img - 1, :]), (0, 0, 1, 0))
        gradienth_genimg_w = F.pad(torch.abs(genimgs[:, :, :, 1:] - genimgs[:, :, :, :w_img - 1]), (1, 0, 0, 0))
        gradient_genimg = gradient_genimg_h + gradienth_genimg_w

        gradient_targetimg_h = F.pad(torch.abs(targetimgs[:, :, 1:, :] - targetimgs[:, :, :h_img - 1, :]), (0, 0, 1, 0))
        gradienth_targetimg_w = F.pad(torch.abs(targetimgs[:, :, :, 1:] - targetimgs[:, :, :, :w_img - 1]), (1, 0, 0, 0))
        gradient_targetimg = gradient_targetimg_h + gradienth_targetimg_w
        
        loss4 = self.criterionL1(gradient_genimg, masks_3c)

        #compute 2rd loss
        loss2 = torch.norm(gradient_genimg / (masks_3c + self.e)) / (channels * h_img * w_img) * math.exp(1 - lamb) * ((stage + 1) * 1.)

        loss1 = torch.sum(torch.mean(torch.minimum((genimgs - targetimgs) ** 2, torch.tensor(lamb + 0.5)), dim=(-1, -2, -3)))
        
        mean_target = torch.mean(targetimgs, dim=(-2, -1), keepdims=True)
        #mean_target = torch.cat([mean_target, mean_target, mean_target], dim=-3)
        if zero:
            loss5 = self.criterionL1(mean_target, genimgs)#, torch.sqrt(torch.tensor(lam)))
            totalloss = self.u*loss2 + self.a*loss4 + loss5
        #compute total loss
        else: totalloss, loss5 = loss1 + self.u*loss2 + self.a*loss4, 0

        return totalloss

