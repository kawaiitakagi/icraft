import torch 
import numpy as np 
import argparse
import matplotlib.pyplot as plt
import sys 
import os
sys.path.append(os.getcwd())
sys.path.append(R'../0_PointNet_seg')

from pointnet.dataset import ShapeNetDataset
from utils.show3d_balls import showpoints
from icraft.xir import *
from icraft.xrt import *
from icraft.host_backend import *
from icraft.buyibackend import *

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default="../3_deploy/modelzoo/PointNet_seg/imodel/8/", help='json & raw path')# pth
parser.add_argument('--idx', type=int, default=0, help='model index')# pred index
parser.add_argument('--dataset', type=str, default='', help='dataset path') 
parser.add_argument('--class_choice', type=str, default='Airplane', help='class choice')# which class you want to seg 
parser.add_argument('--stage',type=str,default='p',help='simulation stage,support p,o,q,a,g')

opt = parser.parse_args()
print(opt)
# ---------------------------------参数设置---------------------------------
STAGE = opt.stage
stage_list = {
    'p':'parsed',
    'o':'optimized',
    'q':'quantized',
    'a':'adapted',
    'g':'BY',
    }
# json & raw 路径设置 
GENERATED_JSON_FILE = opt.model + 'PointNet_seg_' + stage_list[STAGE]+".json"
GENERATED_RAW_FILE = opt.model + 'PointNet_seg_' + stage_list[STAGE]+".raw"

d = ShapeNetDataset(
    root=opt.dataset,
    class_choice=[opt.class_choice],
    split='test',
    data_augmentation=False)
idx = opt.idx
print("Class =",opt.class_choice,"\tSample %d/%d" % (idx, len(d)))
point, seg = d[idx]#input_points, target
point_np = point.numpy()
cmap = plt.cm.get_cmap("hsv", 10)
cmap = np.array([cmap(i) for i in range(10)])[:, :3]
gt = cmap[seg.numpy() - 1, :]
point = point.transpose(1, 0).unsqueeze(0).contiguous()
_data = point.numpy()#(1,3,2500)
print('input size =',_data.shape,type(_data))
input_tensor = Tensor(_data, Layout("**C"))

# 加载指令生成后的网络
generated_network = Network.CreateFromJsonFile(GENERATED_JSON_FILE)
generated_network.loadParamsFromFile(GENERATED_RAW_FILE)
print('INFO: Create network!')

# 创建Session
session = Session.Create([HostBackend],generated_network.view(0),[HostDevice.Default()])
# 应用session
session.apply()

# 模型前向推理
generated_output = session.forward([input_tensor])
# check outputs
# [1,2500,N_CLASS] : pred_results 
# [1,3,3]  : transform_matrix_3x3
# [1,64,64]: transform_matrix_64x64
# for i in range(3):
#     out = np.array(generated_output[i])
#     print(out.shape)
print('INFO: get forward results!')
# 组装成检测结果
outputs = []
for i in range(3):
    out = np.array(generated_output[i])
    print(out.shape)
    outputs.append(torch.tensor(out))
# 后处理
pred_choice = outputs[0].data.max(2)[1]
print('INFO: Get pred results!')
print('pred_result =',pred_choice)

# visualize seg results
pred_color = cmap[pred_choice.numpy()[0], :]
print('INFO: Visualize pred results!')
showpoints(xyz=point_np, c_gt=gt,c_pred=pred_color)