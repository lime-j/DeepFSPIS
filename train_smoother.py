import argparse
import numpy as np

from torch.cuda.amp import autocast as autocast
from models import Adjuster, Smoother
from models.smoother_loss import SmootherLoss
from data.datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch
from trainer import SmootherTrainer
torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=13, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs of training')
parser.add_argument('--workdir', type=str, default='./smoother_training', help='working directory of trainer')
parser.add_argument('--train_dataset_path', type=str, default="/home/lmj/dataset/train2017", help='path of train dataset')
parser.add_argument('--val_dataset_path', type=str, default="/home/lmj/dataset/val2017", help='path of val dataset')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0005, help='adam: learning rate')
parser.add_argument('--decay_epoch', type=int, default=10, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--checkpoint_interval', type=int, default=3, help='interval between saving model checkpoints')
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories

# Losses

criterion = SmootherLoss().cuda()
model = Smoother(inference_only=False).cuda()
adjuster_model = Adjuster().cuda()
criterion = SmootherLoss().cuda()


optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=opt.lr)
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

trainer = SmootherTrainer(
    model,
    adjuster_model=adjuster_model,
    adjuster_path='./checkpoints/adjuster.pth',
    loss = criterion,
    optim = optimizer,
    scheduler= lr_scheduler,
    result_dir=opt.workdir,
    train_dataset_path=opt.train_dataset_path,
    val_dataset_path=opt.val_dataset_path,
    checkpoint_interval=opt.checkpoint_interval,
)

trainer.train()

