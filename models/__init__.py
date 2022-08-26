from .adjuster import Adjuster
from .smoother import Smoother
import torch.nn as nn 
import torch

class DeepFSPIS(nn.Module):
    
    def __init__(self, adjuster_weight_path=None, smoother_weight_path=None):
        super(DeepFSPIS, self).__init__()
        self.smoother = Smoother()
        self.adjuster = Adjuster()
        if adjuster_weight_path is not None:
            self.adjuster.load_state_dict(torch.load(adjuster_weight_path)['icnn'])
        if smoother_weight_path is not None:
            self.smoother.load_state_dict(torch.load(smoother_weight_path)['model'])

    def forward(self, input_image, lamb):
        mask = self.adjuster(input_image, lamb)#.detach()
        generated_images2 = self.smoother(input_image, mask)
        return mask, generated_images2