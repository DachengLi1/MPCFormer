from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import sys
import time
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as dst

from utils import AverageMeter, accuracy, transform_time
from utils import load_pretrained_model, save_checkpoint

from network import mcccnn8

model = mcccnn8(10).cuda()
#print(model.fc.weight[:10])
checkpoint = torch.load("results/base/base-cnn8/checkpoint.pth.tar")
pretrained_dict = checkpoint['net']

model_dict = model.state_dict()
# 1. filter out unnecessary keys
pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict)
# 3. load the new state dict
model.load_state_dict(model_dict)

model.eval()

dataset = dst.CIFAR10
mean = (0.4914, 0.4822, 0.4465)
std  = (0.2470, 0.2435, 0.2616)

test_transform = transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std)
        ])

test_loader = torch.utils.data.DataLoader(
            dataset(root      = "./datasets",
                    transform = test_transform,
                    train     = False,
                    download  = True),
            batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
def test(test_loader, net, criterion):
    losses = AverageMeter()
    top1   = AverageMeter()
    top5   = AverageMeter()

    net.eval()

    end = time.time()
    for i, (img, target) in enumerate(test_loader, start=1):
        img = img.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with torch.no_grad():
            _, _, _, _, _, out = net(img)
            loss = criterion(out, target)

        prec1, prec5 = accuracy(out, target, topk=(1,5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    f_l = [losses.avg, top1.avg, top5.avg]
    logging.info('Loss: {:.4f}, Prec@1: {:.2f}, Prec@5: {:.2f}'.format(*f_l))

    return top1.avg, top5.avg

total_num = 0
total_drop = 0
std = 0
ratio = 9e-3
mean = []
std = []
for name, param in model.named_parameters():
   # size = param.size()
   # total_num += param.numel()
    #print(name, size)
  #  mask = torch.bernoulli(torch.ones(size)*(1-ratio)).cuda()
  #  total_drop += torch.count_nonzero(1-mask)
    param.requires_grad = False
  #  param *= mask
    if "weight" in name and "bn" not in name:
        mean.append(torch.mean(param).item())
        std.append(torch.std(param).item())
    print(f"{name} mean: {torch.mean(param)}, std: {torch.std(param)}")
    # noise = torch.empty(size).normal_(mean=0,std=std).cuda()
    # param += noise
print(f"----------------------total num: {total_num}, drops: {total_drop}------------------------")
print(mean)
print(std)
criterion = torch.nn.CrossEntropyLoss().cuda()
test_top1, test_top5 = test(test_loader, model, criterion)
print(test_top1, test_top5)
