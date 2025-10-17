from __future__ import print_function

import argparse
import numpy as np
import torch
import sys
import os

sys.path.append(os.getcwd())

import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable

import matplotlib.pyplot as plt
sys.path.append(R'../0_PointNet_seg')
from utils.show3d_balls import showpoints
from pointnet.dataset import ShapeNetDataset
#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='../2_compile/fmodel/pointnet_seg_2500x3.pt', help='model path')# pth path
parser.add_argument('--idx', type=int, default=0, help='model index')# pred index
parser.add_argument('--dataset', type=str, default='', help='dataset absolute path') 
parser.add_argument('--class_choice', type=str, default='Airplane', help='class choice')# which class you want to seg 

opt = parser.parse_args()
print(opt)

d = ShapeNetDataset(
    root=opt.dataset,
    class_choice=[opt.class_choice],
    split='test',
    data_augmentation=False)

idx = opt.idx

print("Class =",opt.class_choice,"\tSample %d/%d" % (idx, len(d)))
point, seg = d[idx]#input_points, target
print("input_size=",point.size(), "target size =",seg.size())

point_np = point.numpy()

cmap = plt.cm.get_cmap("hsv", 10)
cmap = np.array([cmap(i) for i in range(10)])[:, :3]
gt = cmap[seg.numpy() - 1, :]
# prepare input 
point = point.transpose(1, 0).contiguous()
print('Point =',point.shape,type(point))
point = Variable(point.view(1, point.size()[0], point.size()[1]))
# 加载traced模型
print("**** =",opt.model)
print(os.path.abspath(opt.model))

model = torch.jit.load(opt.model)
pred, _, _ = model(point)
print('INFO: Network load Done!')
# pred, _, _ = classifier(point)
pred_choice = pred.data.max(2)[1]
print('INFO: Get pred results!')
print('pred_result =',pred_choice)

#print(pred_choice.size())
pred_color = cmap[pred_choice.numpy()[0], :]
print('INFO: Visualize pred results!')
#print(pred_color.shape)
print('GT =',gt)
print('*'*80)
print('pred =',pred_color)
showpoints(point_np, gt, pred_color)
