import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import random

import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import DeepFSPIS
from data.datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default="smoothing_result", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=4, help='size of the batches')
parser.add_argument('--resize_image', type=bool, default=False, help="resize image to desired size")
parser.add_argument('--img_height', type=int, default=480, help='size of image height')
parser.add_argument('--img_width', type=int, default=640, help='size of image width')
parser.add_argument("--checkpoints_dir", type=str, default="checkpoints", help="models are saved here")
parser.add_argument("--img_dir", type=str, default="./test_image", help="path to image dir")
parser.add_argument("--lamb", type=str, default="0.6,0.5,0.4", help="strength of smoothing")
opt = parser.parse_args()

save_dir = opt.save_dir
os.makedirs(save_dir, exist_ok=True)

if torch.cuda.is_available() : 
    torch.backends.cudnn.benchmark = True
    cuda = True
else :
    cuda = False

model = DeepFSPIS(adjuster_weight_path=os.path.join(opt.checkpoints_dir, "adjuster.pth"),
                  smoother_weight_path=os.path.join(opt.checkpoints_dir, "smoother.pth"))

if cuda:
    model = model.to("cuda")
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor


# Image transformations
if opt.resize_image :
    transforms_ = [transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                   transforms.ToTensor()]
else :
    transforms_ = [transforms.ToTensor()]

val_dataloader = DataLoader(ImageDataset(opt.img_dir, cache=True, transforms_=transforms_, unaligned=True), 
                            batch_size=opt.batch_size, shuffle=False, num_workers=8)

ups = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
lamb_lst = map(float, opt.lamb.strip().split(","))
lamb_lst = [torch.tensor(it) for it in lamb_lst]
cnt = 0;

def to_image(img, suffix, lamb):
    ndarr = img[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(save_dir + '/%s_%.3f.png' % (suffix, lamb))

def to_same_name(img, suffix, lamb):
    ndarr = img[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(save_dir + '/%s.png' % (suffix))

model.eval()

with torch.no_grad():
    cur = 0;
    for i, batch in enumerate(val_dataloader):
        # Set model input
        input_image = batch['img'].to("cuda") 
        for lamb in lamb_lst:
            mask_images, generated_images = model(input_image, lamb)
            mask = torch.cat([mask_images, mask_images, mask_images], dim=1)
            to_image(mask.data, "%s_%s" % ("mask", batch["fn"][0].split('"')[0]), lamb)
            to_image(generated_images.data, "%s_%s" % ("gen", batch["fn"][0].split('"')[0]), lamb)

