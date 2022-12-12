import argparse
import numpy as np

from cosine_ann import CosineAnnealingWarmupRestarts
from models import Adjuster, Smoother
from models.adjuster_loss import get_adjuster_loss
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch
from trainer import AdjusterTrainer

torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs of training')
parser.add_argument('--workdir', type=str, default='./smoother_training', help='working directory of trainer')
parser.add_argument('--train_dir', type=str, default="/home/lmj/dataset/train2017", help='path of train dataset')
parser.add_argument('--edge_dir', type=str, default="/home/lmj/dataset/train2017", help='path of train dataset')
parser.add_argument('--val_dataset_path', type=str, default="/home/lmj/dataset/val2017", help='path of val dataset')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--checkpoint_interval', type=int, default=3, help='interval between saving model checkpoints')
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories

# Losses

criterion = get_adjuster_loss()
model = Adjuster().cuda()



optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=opt.lr)
lr_scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=20, max_lr=opt.lr, warmup_steps=1, min_lr=1e-6, ratio=1)
trainer = AdjusterTrainer(
    model,
    loss = criterion,
    optim = optimizer,
    scheduler= lr_scheduler,
    train_dir=opt.train_dir,
    edge_dir=opt.edge_dir,
    checkpoint_interval=opt.checkpoint_interval,
    num_threads=opt.n_cpu
)

trainer.train()


