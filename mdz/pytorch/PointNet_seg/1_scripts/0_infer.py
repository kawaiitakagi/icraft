from __future__ import print_function
import argparse
import numpy as np
import torch
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(R'../0_PointNet_seg')

from utils.show3d_balls import showpoints

import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetDenseCls
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='../weights/seg_model_Airplane_4.pth', help='model path')# pth
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

state_dict = torch.load(opt.model)
classifier = PointNetDenseCls(k= state_dict['conv4.weight'].size()[0],feature_transform=True)#use feature transformation or not
# params = classifier.state_dict()#获得模型原始状态以及参数
# for k,v in params.items():
#     print(k)#只打印k值 

classifier.load_state_dict(state_dict)
classifier.eval()
print('INFO: Network load Done!')
point = point.transpose(1, 0).contiguous()

point = Variable(point.view(1, point.size()[0], point.size()[1]))
pred, _, _ = classifier(point)
print('pred =',pred,pred.shape)
pred_choice = pred.data.max(2)[1]
print('INFO: Get pred results!')
print('pred_result =',pred_choice)

#print(pred_choice.size())
pred_color = cmap[pred_choice.numpy()[0], :]
print('INFO: Visualize pred results!')
#print(pred_color.shape)
showpoints(point_np, gt, pred_color)
