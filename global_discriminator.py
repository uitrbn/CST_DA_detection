import torch.nn as nn
import random
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function
import torchvision.models as models
import numpy as np
import time
import pdb
import cv2
from skimage.transform import resize
import scipy
from scipy.misc import imresize

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
        
class LocalDiscriminator(nn.Module):
    def __init__(self, in_channels):
        super(LocalDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        x = input
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        context_vector = x.detach()
        x = self.conv3(x)
        x = F.sigmoid(x)
        return x, context_vector
        
class LossForLocal(nn.Module):
    def __init__(self):
        super(LossForLocal, self).__init__()

    def forward(self, input, domain, heat_map=None):
        y = input
        if domain==1:
            loss = y.pow(2)
        elif domain==0:
            loss = (1-y).pow(2)
        else:
            raise Exception("domain label must be consistent in one batch")
        if heat_map is None:
            return loss
        else:
            heat_map = heat_map[0][9:, :, :].detach()
            heat_map = heat_map.sum(0).cpu().view(heat_map.shape[1], heat_map.shape[2]).numpy()
            heat_map = resize(heat_map, (loss.shape[2], loss.shape[3]))

            return loss * torch.FloatTensor(heat_map).view(1, 1, heat_map.shape[0], heat_map.shape[1]).cuda()


class LossForRPNCLS(nn.Module):
    def __init__(self, power):
        super(LossForRPNCLS, self).__init__()
        self.power = power
    
    def forward(self, cls_score, roi_score):
        factor_bigger_half = (2 * roi_score - 1).pow(self.power)
        factor_smaller_half = (1 - 2 * roi_score).pow(self.power)
        factor = (roi_score > 0.5).float() * factor_bigger_half + (roi_score <= 0.5).float() * factor_smaller_half
        
        factor = factor.detach()

        ent = F.softmax(cls_score, dim=1) * F.log_softmax(cls_score, dim=1)
        ent = -1.0 * ent.sum(dim=1)
        return ent * factor

class LossForDiscrepancy(nn.Module):
    def __init__(self):
        super(LossForDiscrepancy, self).__init__()
    
    def forward(self, cls_score, roi_score):
        cls_prob = F.softmax(cls_score, 1)
        weight = torch.min( 2 * torch.min(1- cls_prob[:, 0], cls_prob[:, 0]), 2* torch.min(roi_score[:], 1 - roi_score[:]))
        weight = weight.detach()
        weight = weight.pow(2)
        loss_to_min = (1 - cls_prob[:, 0] - roi_score[:]).abs() * weight
        loss_to_max = -(1 - cls_prob[:, 0] - roi_score[:]).abs() * weight
        return loss_to_min, loss_to_max