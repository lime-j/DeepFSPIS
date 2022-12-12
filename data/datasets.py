import glob
import random
import os

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from PIL import ImageFile
to_tensor = transforms.Compose([transforms.ToTensor()])
from utils import make_dataset
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True
from typing import Dict, AnyStr, Any

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

    
class COCOHIPeDataset(Dataset):
    """

    """
    def __init__(self, origin_dir, edge_dir, syn_dir = None, data_len=None, thresh=0.3, stages=4):
        super(COCOHIPeDataset, self).__init__()
        print(data_len)
        self.origin_dir = origin_dir
        self.edge_dir = edge_dir
        self.fns = None
        self.thresh = thresh
        sortkey = lambda key: os.path.split(key)[-1]
        self.origin_path = sorted(make_dataset(self.origin_dir), key=sortkey)
        self.origin_size = len(self.origin_path)
        self.stages = stages
        self.syn_dir = syn_dir
        if data_len is None:
            self.size = len(self.origin_path)
        else:
            self.size = min(data_len, len(self.origin_path))

    def __len__(self):
        return self.size

    def __getitem__(
            self,
            index: int
    ) -> Dict[AnyStr, Any]:
        if index > self.size:
            raise IndexError("out of the range")
        paths = self.origin_path[index]
        
        org_input = Image.open(paths).convert("RGB")##.resize((256, 256))
        w, h = org_input.size
        org_input = org_input.resize((480, 640)) #((int(w / 1.5), int(h / 1.5)))
        ret = {
            "org_input": to_tensor(org_input),
            "fn": self.origin_path[index % self.origin_size].split('/')[-1].rstrip('.jpg'),
            "edge" : [],
            "mask" : []
        }
        for idx in range(self.stages):
            edge_input = Image.open(paths.replace(self.origin_dir, self.edge_dir).replace(".jpg", "_%d.png" % idx)).convert("L").resize((480, 640)) #((int(w / 1.5), int(h / 1.5)))
            #.resize((256, 256))
            label = np.array(edge_input, dtype=np.float32)[..., np.newaxis].transpose(2, 0, 1)
            label[label < 0.5] = 0
            #label[np.logical_and(label > 0, label < self.thresh * 255)] = 2
            label[label > 0.5] = 1
            ret["edge"].append(to_tensor(edge_input))
            ret["mask"].append(torch.from_numpy(label))
        if self.syn_dir is not None:
            for idx in range(10):
                try :
                    edge_input = Image.open(
                        paths.replace(self.origin_dir, self.syn_dir).replace(".jpg", "_3_syn_%d.png" % idx)).convert("L").resize((int(w / 1.2), int(h / 1.2)))
                except FileNotFoundError:
                    break
                else :
                    label = np.array(edge_input, dtype=np.float32)[..., np.newaxis].transpose(2, 0, 1)
                    label[label < 0.5] = 0
                    # label[np.logical_and(label > 0, label < self.thresh * 255)] = 2
                    label[label > 0.5] = 1
                    ret["edge"].append(to_tensor(edge_input))
                    ret["mask"].append(torch.from_numpy(label))

        return ret
