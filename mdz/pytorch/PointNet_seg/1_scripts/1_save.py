from __future__ import print_function

import argparse
import numpy as np
import torch
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(R'../0_PointNet_seg')

import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from pointnet.dataset import ShapeNetDataset
from model_for_export import PointNetDenseCls#这里变了 要注意 不是model!!!!
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='../weights/seg_model_Airplane_4.pth', help='model path')# pth
parser.add_argument('--idx', type=int, default=0, help='model index')# pred index
parser.add_argument('--dataset', type=str, default='', help='dataset path') 
parser.add_argument('--class_choice', type=str, default='Airplane', help='class choice')# which class you want to seg 
parser.add_argument('--dst', type=str, default= '../2_compile/fmodel', help='traced model path')
parser.add_argument('--name',type=str,default= 'pointnet_seg_2500x3.pt', help='traced model name')
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
print('***opt.model =',opt.model)
state_dict = torch.load(opt.model)
# fstn.fc3 融 add操作 
_iden = Variable(torch.from_numpy(np.eye(64).flatten().astype(np.float32))).view(1,4096).repeat(1,1).squeeze(0)
state_dict['feat.fstn.fc3.bias'] = state_dict['feat.fstn.fc3.bias']+_iden

classifier = PointNetDenseCls(k= state_dict['conv4.weight'].size()[0],feature_transform=True)#use feature transformation or not
# params = classifier.state_dict()#获得模型原始状态以及参数
# for k,v in params.items():
#     print(k)#只打印k值 
# exit()
classifier.load_state_dict(state_dict)
classifier.eval()
print('INFO: Network load Done!')

# prepare input 
point = point.transpose(1, 0).contiguous()

point = Variable(point.view(1, point.size()[0], point.size()[1]))
print('input size =',point.shape)

# export traced_model
traced_model = torch.jit.trace(classifier,point,strict=False)
print('*'*20,'Export & Save Traced_model','*'*20)
# create folder
try:
    os.makedirs(opt.dst)
    print(f"The directory {opt.dst} has been successfully created.")
except FileExistsError:
    print(f"The directory {opt.dst} already exists.")
except OSError as error:
    print(f"The directory could not be created: {error}")
TRACE_PATH = opt.dst + "/" + opt.name
torch.jit.save(traced_model,TRACE_PATH)
print('TorchScript export success, saved in %s' % TRACE_PATH)
print('*'*20,'Export & Save Done!','*'*20)