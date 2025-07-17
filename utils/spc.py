import os
import sys
import shutil
import random
import logging
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from dataloaders import BaseDataSets, TwoStreamBatchSampler
from dataloaders.transform import RandomGenerator, WeakStrongAugment
from networks import net_factory
from . import losses, val_2d, config, plot
from . import patients_to_slices, get_current_consistency_weight, update_ema_variables

def style_mixing_uniform(image_batch, ranges=(0, 1)):
    """Apply style mixing to the input image batch."""
    batch_size = image_batch.shape[0]
    mu = image_batch.mean(dim=(-2, -1), keepdims=True)
    sig = (image_batch.var(dim=(-2, -1), keepdims=True) + 1e-6).sqrt()
    normed_batch = (image_batch - mu) / sig

    lmda = np.random.uniform(ranges[0], ranges[1], (batch_size, 1, 1, 1))
    lmda = torch.from_numpy(lmda).to(image_batch.device).float()

    perm = torch.arange(batch_size - 1, -1, -1)
    perm_j, perm_i = torch.chunk(perm, 2)
    perm_j = perm_j[torch.randperm(batch_size // 2)]
    perm_i = perm_i[torch.randperm(batch_size // 2)]
    perm = torch.cat([perm_j, perm_i], dim=0)

    mu2, sig2 = mu[perm], sig[perm]
    mu_mix = lmda * mu + (1 - lmda) * mu2
    sig_mix = lmda * sig + (1 - lmda) * sig2
    return torch.clamp(normed_batch * sig_mix + mu_mix, 0, 1)

class FeatureMemory:
    """A class to maintain a memory of features for each class."""

    def __init__(self, max_size, num_classes, feat_dim=16):
        self.max_size = max_size
        self.num_classes = num_classes
        self.protos = torch.zeros((num_classes, max_size, feat_dim)).float().cuda()
        self.weights = torch.zeros((num_classes, max_size)).float().cuda()
        self.pointers = torch.zeros(num_classes).long().cuda()

        self.sigma = 2
        self.m = 0.999
        self.avg_ema = torch.ones(num_classes).cuda() * 1.0 / num_classes
        self.var_ema = torch.ones(num_classes).cuda()

    @torch.no_grad()
    def enqueue(self, prob, feat):
        """Enqueue features and probabilities into the memory."""
        num_classes = prob.shape[1]
        pred = torch.argmax(prob, dim=1)  # [b, h, w]
        for c in range(num_classes):
            pred_c = pred.eq(c).float().unsqueeze(1)  # [b, 1, h, w]
            pred_c = pred_c.sum(dim=[2, 3]).any(dim=1, keepdims=True).float()  # [b, 1]

            prob_c = prob[:, c, :, :].unsqueeze(1)  # [b, 1, h, w]
            feat_c = (prob_c * feat).sum(dim=[2, 3])  # [b, d]
            feat_c = feat_c / (prob_c.sum(dim=[2, 3]) + 1e-8)  # [b, d]

            feat_c = (feat_c * pred_c).sum(dim=0) / (pred_c.sum() + 1e-8)  # [d]
            weight_c = prob_c.mean(dim=[1, 2, 3])
            weight_c = (weight_c * pred_c).sum() / (pred_c.sum() + 1e-8)  # [1]

            # enqueue
            self.protos[c, self.pointers[c], :] = feat_c
            self.weights[c, self.pointers[c]] = weight_c
            self.pointers[c] = (self.pointers[c] + 1) % self.max_size

    def calc_protos(self):
        """Calculate the prototypes for each class."""
        protos = (self.protos * self.weights.unsqueeze(2)).sum(dim=1)  # [c, d]
        return protos / (self.weights.sum(dim=1, keepdims=True) + 1e-8)
