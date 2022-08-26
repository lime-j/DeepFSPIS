import glob
import random
import os

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from PIL import ImageFile

from utils import make_dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageDataset(Dataset):

    @staticmethod
    def process_file(img_path, transform, upsample_ratio=None, downsample_ratio=None):
        item_img = Image.open(img_path).convert("RGB")
        w, h = item_img.size

        if  upsample_ratio is not None:
            item_img = item_img.resize((w * upsample_ratio, h * upsample_ratio))
        elif downsample_ratio is not None:
            item_img = item_img.resize((w // downsample_ratio, h // downsample_ratio))
        w, h = item_img.size
        if w % 4 or h % 4: item_img = item_img.resize((w + 4 - w % 4, h + 4 - h % 4))
        
        item_img = transform(item_img)

        return {'img': item_img, 'fn': img_path.split('/')[-1].rstrip(".jpg")}
    
    def prepare_file(self, index):
        return ImageDataset.process_file(self.origin_path[index + self.delta], self.transform, self.upsample, self.downsample)


    def index_file(self, index): return self.img_set[index]

    def id(self, index) : return index

    def __init__(self, root, cache=False, transforms_=None, unaligned=False, delta=None, size=None, upsample=None, downsample=None):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.root = root
        self.origin_path = sorted(make_dataset(root))

        if size is None : self.len = len(self.origin_path)
        else : self.len = size
        self.upsample = upsample
        self.downsample = downsample
        self.delta = delta if delta is not None else 0

        if cache : self.hdl, self.get_item = self.prepare_file, self.index_file
        else : self.hdl, self.get_item = self.id, self.prepare_file
        self.img_set = [self.hdl(index) for index in range(self.len)]

    
    def __getitem__(self, index) : return self.get_item(index)

    def __len__(self) : return self.len
