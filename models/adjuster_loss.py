import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage


def compute_gradient(img):
    gradx = img[..., 1:, :] - img[..., :-1, :]
    grady = img[..., 1:] - img[..., :-1]
    return gradx, grady

class GradientExclusiveLoss(nn.Module):
    def __init__(self):
        super(GradientExclusiveLoss, self).__init__()

    def forward(self, predict, target):
        predict_x = F.pad(torch.abs(predict[:, :, :, :-1] - predict[:, :, :, 1:]), (1, 0, 0, 0))
        predict_y = F.pad(torch.abs(predict[:, :, :-1, :] - predict[:, :, 1:, :]), (0, 0, 1, 0))
        target_x = F.pad(torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:]), (1, 0, 0, 0))
        target_y = F.pad(torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :]), (0, 0, 1, 0))
        predict_d = (predict_x + predict_y) / 2
        target_d = (target_x + target_y) / 2
        return torch.mean(predict_d * target_d)

    def get_loss(self, predict, target): return self.forward(predict, target)


class SquaredGradientExclusiveLoss(nn.Module):
    def __init__(self): super(SquaredGradientExclusiveLoss).__init__()

    def get_loss(self, predict, target):
        predict_d, target_d = edge_calc_sqrt(predict), edge_calc_sqrt(target)
        return torch.mean(torch.sqrt(predict_d * target_d))


def edge_calc(smoothed_map):
    edg_x = F.pad(torch.abs(smoothed_map[:, :, :, :-1] - smoothed_map[:, :, :, 1:]), (1, 0, 0, 0))
    edg_y = F.pad(torch.abs(smoothed_map[:, :, :-1, :] - smoothed_map[:, :, 1:, :]), (0, 0, 1, 0))
    return (edg_x + edg_y) / 2


def edge_calc_sqrt(smoothed_map):
    edg_x = F.pad(torch.abs(smoothed_map[:, :, :, :-1] - smoothed_map[:, :, :, 1:]), (1, 0, 0, 0))
    edg_y = F.pad(torch.abs(smoothed_map[:, :, :-1, :] - smoothed_map[:, :, 1:, :]), (0, 0, 1, 0))
    return torch.sqrt((edg_x * edg_x + edg_y * edg_y) / 2)


class EdgeConsistLoss(nn.Module):
    r"""
    :math: \nabla S_{\lambda} \cdot \nabla S_{\lambda - \epsilon} shall equals to \nabla S_{lambda} for all \epslion \le \lambda

    Args :
        - center : smoothed image at \lambda
        - center_min : smoothed image at \lambda - \epsilon_0
        - center_add : smoothed image at \lambda + \epsilon_1
    """

    def __init__(self):
        super(EdgeConsistLoss, self).__init__()
        self.loss = lambda it: torch.mean(torch.abs(it))
        self.mse = nn.MSELoss()
        self.eps = 1e-2

    def forward(self, center, center_min):
        center, center_min = edge_calc(center), edge_calc(center_min)
        return self.mse(center, torch.sqrt(center * center_min))


class PreservConsistLoss(nn.Module):
    def __init__(self):
        super(PreservConsistLoss, self).__init__()
        self.loss = lambda it: torch.mean(torch.abs(it))
        self.criterion = nn.MSELoss()
        self.eps = 1e-2

    def forward(self, predict, target, original):
        # print(original.shape)
        predict_x = F.pad(torch.abs(predict[:, :, :, :-1] - predict[:, :, :, 1:]), (1, 0, 0, 0))
        predict_y = F.pad(torch.abs(predict[:, :, :-1, :] - predict[:, :, 1:, :]), (0, 0, 1, 0))
        target_x = F.pad(torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:]), (1, 0, 0, 0))
        target_y = F.pad(torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :]), (0, 0, 1, 0))
        original_x = F.pad(torch.abs(original[:, :, :, :-1] - original[:, :, :, 1:]), (1, 0, 0, 0))
        original_y = F.pad(torch.abs(original[:, :, :-1, :] - original[:, :, 1:, :]), (0, 0, 1, 0))
        # target_gradx[target_gradx < self.eps] = self.eps
        # target_grady[target_grady < self.eps] = self.eps
        predict_d = (predict_x + predict_y) / 2
        target_d = (target_x + target_y) / 2
        # target_d[target_d < 1e-2] = 0
        original_d = (original_x + original_y) / 2

        target_d, _ = torch.max(target_d, dim=1)
        target_d.unsqueeze(1)  # .unsqueeze(1)
        original_d, _ = torch.max(original_d, dim=1)
        original_d.unsqueeze(1)  # .unsqueeze(1)

        target_d = torch.cat([target_d, target_d, target_d], dim=0).unsqueeze(0)
        target_d_avg = torch.mean(target_d)
        target_d[target_d < target_d_avg] = 0
        original_d = torch.cat([original_d, original_d, original_d], dim=0).unsqueeze(0)
        # print(target_d.shape, original_d.shape)
        alpha = 0
        target_final = (target_d * alpha + (1 - alpha) * torch.sqrt(target_d * original_d))
        loss1 = self.criterion(predict_d, target_final)
        loss2 = self.loss(predict_d / (target_final + self.eps))
        return 0.5 * loss1 + 4 * loss2


class SquaredPreservConsistLoss(nn.Module):
    def __init__(self):
        super(SquaredPreservConsistLoss, self).__init__()
        self.loss = lambda it: torch.mean(torch.abs(it))
        self.criterion = nn.MSELoss()
        self.eps = 1e-2

    def forward(self, predict, target, original):
        predict_d, target_d, original_d = edge_calc_sqrt(predict), edge_calc_sqrt(target), edge_calc_sqrt(original)
        target_d = torch.cat([target_d, target_d, target_d], dim=1)
        original_d = torch.cat([original_d, original_d, original_d], dim=1)
        alpha = 0.5
        target_final = (target_d * alpha + (1 - alpha) * torch.sqrt(target_d * original_d))
        loss1 = self.criterion(predict_d, target_final)
        loss2 = self.loss(predict_d / (target_final + self.eps))
        return 0.4 * loss1 + 4 * loss2


class GradientPenaltyLoss(nn.Module):
    def __init__(self):
        super(GradientPenaltyLoss, self).__init__()
        self.loss = lambda it: torch.mean(torch.abs(it))
        self.criterion = nn.MSELoss()
        self.eps = 1e-3

    def forward(self, predict, target, ratio, calib=None):
        pred_cnt = torch.sum(predict, dim=(-2, -1))
        gt_cnt = torch.sum(target, dim=(-2, -1))
        # print(pred_cnt.shape, gt_cnt.shape, ratio.shape)
        # print(pred_cnt.shape, gt_cnt.shape, ratio.shape)
        if calib is None:
            loss2 = self.loss(pred_cnt / (gt_cnt + self.eps) - ratio)
        else:
            loss2 = self.loss(calib(pred_cnt / (gt_cnt + self.eps)) - ratio)
        return loss2


class GradientConsistencyLoss(nn.Module):
    def __init__(self):
        super(GradientConsistencyLoss, self).__init__()
        self.loss = lambda it: torch.mean(torch.abs(it))
        self.beta = 0.5

    def forward(self, predict, target, gam, lonly):
        if lonly:
            loss = self.loss(predict * (1 - target)) #+ self.beta * self.loss(target * gam * (1 - predict))
        else:
            loss = self.loss(predict * (1 - target)) + self.loss(target * (1 - predict))
        return loss


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, predict, target):
        predict_gradx, predict_grady = compute_gradient(predict)
        target_gradx, target_grady = compute_gradient(target)

        return self.loss(predict_gradx, target_gradx) + self.loss(predict_grady, target_grady)


class RCFCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(RCFCrossEntropyLoss, self).__init__()

    def forward(self, prediction, labelf, beta=1.0):
        label = labelf.long()
        mask = labelf.clone()
        num_positive = torch.sum(label == 1).float()
        num_negative = torch.sum(label == 0).float()

        mask[label == 1] = 1.0 #* num_negative / (num_positive + num_negative)
        mask[label == 0] = 1.0 #beta * num_positive / (num_positive + num_negative)
        mask[label == 2] = 0
        if num_positive == 0:
            mask[label == 0] = 1.0
            mask[label == 1] = 1.0
        # print(num_positive, num_negative)
        cost = F.binary_cross_entropy(prediction, labelf, weight=mask, reduction='sum')

        return cost / (prediction.shape[-1] * prediction.shape[-2] * prediction.shape[-3])


class DiceLoss(nn.Module):
    """
    Implements re_dice loss from paper "Learning to Predict Crisp Boundaries", ECCV 2018
    """

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, predict, target):
        u = torch.sum(predict ** 2, dim=0) + torch.sum(target ** 2, dim=0)
        d = torch.sum(2 * predict * target, dim=0)
        return torch.mean(u / (d + 1e-3))

class MultipleLoss(nn.Module):
    def __init__(self, losses, weight=None):
        super(MultipleLoss, self).__init__()
        self.losses = nn.ModuleList(losses)
        self.weight = weight or [1 / len(self.losses)] * len(self.losses)

    def forward(self, predict, target):
        total_loss = 0
        for weight, loss in zip(self.weight, self.losses):
            total_loss += loss(predict, target) * weight
        return total_loss


def l1_norm_dim(x, dim):
    return torch.mean(torch.abs(x), dim=dim)


def l1_norm(x):
    return torch.mean(torch.abs(x))


def l2_norm(x):
    return torch.mean(torch.square(x))


def gradient_norm_kernel(x, kernel_size=10, gaussian_kernel=None):
    out_h, out_v = compute_gradient(x)
    shape = out_h.shape
    out_h = F.unfold(out_h, kernel_size=(kernel_size, kernel_size), stride=(1, 1))
    out_h = out_h.reshape(shape[0], shape[1], kernel_size * kernel_size, -1)
    out_h = l1_norm_dim(out_h, 2)
    out_v = F.unfold(out_v, kernel_size=(kernel_size, kernel_size), stride=(1, 1))
    out_v = out_v.reshape(shape[0], shape[1], kernel_size * kernel_size, -1)

    out_v = l1_norm_dim(out_v, 2)
    if gaussian_kernel is None:
        return out_h, out_v
    else:
        return out_h * gaussian_kernel, out_v * gaussian_kernel


def norm_kernel(grad_h, grad_v, kernel_size=30):
    shape = grad_h.shape
    out_h = F.unfold(grad_h, kernel_size=(kernel_size, kernel_size), stride=(1, 1))
    out_h = out_h.reshape(shape[0], shape[1], kernel_size * kernel_size, -1)
    out_h = out_h.transpose(-1, -2)
    #out_h = l1_norm_dim(out_h, 2)
    out_v = F.unfold(grad_v, kernel_size=(kernel_size, kernel_size), stride=(1, 1))
    out_v = out_v.reshape(shape[0], shape[1], kernel_size * kernel_size, -1)
    out_v = out_v.transpose(-1, -2)
    #out_v = l1_norm_dim(out_v, 2)
    return out_h, out_v

class RTVLoss(nn.Module):

    def __init__(self, kernel_size=5, sigma=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.criterion = nn.L1Loss()
        self.eps = 1e-6
        self.gaussian_kernel = (torch.from_numpy(scipy.ndimage.gaussian_filter(kernel_size, sigma=sigma)) * math.sqrt(
            2 * math.pi) * sigma * sigma).to("cuda")
        self.gaussian_kernel = self.gaussian_kernel.reshape(-1)

    def forward(self, grad_h, grad_w, gam):
        out_r_normx, out_r_normy = norm_kernel(grad_h, grad_w, self.kernel_size)
        lx, ly = torch.abs(self.gaussian_kernel * torch.sum(out_r_normx, dim=-1)),\
                 torch.abs(self.gaussian_kernel * torch.sum(out_r_normy, dim=-1))
        dx, dy = torch.sum(self.gaussian_kernel * torch.abs(out_r_normx), dim=-1),\
                 torch.sum(self.gaussian_kernel * torch.abs(out_r_normy), dim=-1)
        #print(lx.shape, dx.shape, ly.shape, dy.shape)
        norm_loss = (dx / (lx + self.eps) * (torch.exp(1 - gam))).mean() + (dy / (ly + self.eps) * (torch.exp(1 - gam))).mean()
        return norm_loss


class ContentLoss():
    def initialize(self, loss):
        self.criterion = loss

    def get_loss(self, fakeIm, realIm):
        return self.criterion(fakeIm, realIm)


def get_adjuster_loss():
    loss_dic = {}

    t_pixel_loss = GradientPenaltyLoss()  # ], [1.0]))
    t_fidelity_loss = ContentLoss()
    t_fidelity_loss.initialize(MultipleLoss([nn.L1Loss()], [1.0]))

    loss_dic['t_fidelity'] = RCFCrossEntropyLoss()
    loss_dic['t_reduction'] = t_pixel_loss
    loss_dic['t_consistency'] = GradientConsistencyLoss()
    loss_dic['t_dice'] = DiceLoss()
    return loss_dic


if __name__ == '__main__':
    x = torch.randn(3, 32, 224, 224).cuda()
    import time

    s = time.time()
    out1, out2 = gradient_norm_kernel(x)
    t = time.time()
    print(t - s)
    print(out1.shape, out2.shape)
