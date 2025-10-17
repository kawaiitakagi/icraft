"""
 Copyright 2023  FMSH.Co.Ltd. All rights reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import sys
sys.path.append(R"../0_yolov7_pose")
import matplotlib.pyplot as plt
import torch
import cv2
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression
from utils.plots import output_to_target,plot_skeleton_kpts
from models.yolo import *
import types

#----------------------------
# 路径配置
#----------------------------
# 原模型地址
W_PATH = R"../weights/yolov7-w6-pose-decouple.pt"
# trace后模型地址
TRACE_PATH = R"../2_compile/fmodel/yolov7-w6-pose-decouple-640x640.pt"
# 输入图片地址
IMG_PATH = R'../0_yolov7_pose/onnx_inference/img.png'

#----------------------------
# 获取trace模型推理结果
#----------------------------
model = torch.jit.load(TRACE_PATH,map_location="cpu")
_ = model.float().eval()
image = cv2.imread(IMG_PATH)
image = letterbox(image, 640, stride=64, auto=True)[0]
image_ = image.copy()
image = transforms.ToTensor()(image)
image = torch.tensor(np.array([image.numpy()]))
TRACE_OUTPUT = model(image)

#----------------------------
# 将结果送入原模型裁掉部分计算
#----------------------------
weigths = torch.load(W_PATH, map_location="cpu")
dynm_model = weigths['model'].float().eval()

def after_detconv(self,x):
    z = []  # inference output
    self.training |= self.export
    for i in range(self.nl):
        x[i] = torch.cat((TRACE_OUTPUT[2*i], TRACE_OUTPUT[2*i+1]), axis=1)
        bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        _x_det = x[i][:,:self.no_det*self.na,:,:].view(bs, self.na, self.no_det, ny, nx)
        _x_kpt = x[i][:,self.no_det*self.na:,:,:].view(bs, self.na, self.no_kpt, ny, nx)
        x[i] = torch.cat((_x_det, _x_kpt), axis=2).permute(0, 1, 3, 4, 2).contiguous()
        x_det = x[i][..., :6]
        x_kpt = x[i][..., 6:]

        if not self.training:  # inference
            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
            kpt_grid_x = self.grid[i][..., 0:1]
            kpt_grid_y = self.grid[i][..., 1:2]

            if self.nkpt == 0:
                y = x[i].sigmoid()
            else:
                y = x_det.sigmoid()

            if self.inplace:
                xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2) # wh
                if self.nkpt != 0:
                    x_kpt[..., 0::3] = (x_kpt[..., ::3] * 2. - 0.5 + kpt_grid_x.repeat(1,1,1,1,17)) * self.stride[i]  # xy
                    x_kpt[..., 1::3] = (x_kpt[..., 1::3] * 2. - 0.5 + kpt_grid_y.repeat(1,1,1,1,17)) * self.stride[i]  # xy
                    #x_kpt[..., 0::3] = (x_kpt[..., ::3] + kpt_grid_x.repeat(1,1,1,1,17)) * self.stride[i]  # xy
                    #x_kpt[..., 1::3] = (x_kpt[..., 1::3] + kpt_grid_y.repeat(1,1,1,1,17)) * self.stride[i]  # xy
                    #print('=============')
                    #print(self.anchor_grid[i].shape)
                    #print(self.anchor_grid[i][...,0].unsqueeze(4).shape)
                    #print(x_kpt[..., 0::3].shape)
                    #x_kpt[..., 0::3] = ((x_kpt[..., 0::3].tanh() * 2.) ** 3 * self.anchor_grid[i][...,0].unsqueeze(4).repeat(1,1,1,1,self.nkpt)) + kpt_grid_x.repeat(1,1,1,1,17) * self.stride[i]  # xy
                    #x_kpt[..., 1::3] = ((x_kpt[..., 1::3].tanh() * 2.) ** 3 * self.anchor_grid[i][...,1].unsqueeze(4).repeat(1,1,1,1,self.nkpt)) + kpt_grid_y.repeat(1,1,1,1,17) * self.stride[i]  # xy
                    #x_kpt[..., 0::3] = (((x_kpt[..., 0::3].sigmoid() * 4.) ** 2 - 8.) * self.anchor_grid[i][...,0].unsqueeze(4).repeat(1,1,1,1,self.nkpt)) + kpt_grid_x.repeat(1,1,1,1,17) * self.stride[i]  # xy
                    #x_kpt[..., 1::3] = (((x_kpt[..., 1::3].sigmoid() * 4.) ** 2 - 8.) * self.anchor_grid[i][...,1].unsqueeze(4).repeat(1,1,1,1,self.nkpt)) + kpt_grid_y.repeat(1,1,1,1,17) * self.stride[i]  # xy
                    x_kpt[..., 2::3] = x_kpt[..., 2::3].sigmoid()

                y = torch.cat((xy, wh, y[..., 4:], x_kpt), dim = -1)

            else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                if self.nkpt != 0:
                    y[..., 6:] = (y[..., 6:] * 2. - 0.5 + self.grid[i].repeat((1,1,1,1,self.nkpt))) * self.stride[i]  # xy
                y = torch.cat((xy, wh, y[..., 4:]), -1)

            z.append(y.view(bs, -1, self.no))

    return x if self.training else (torch.cat(z, 1), x)

IKeypoint.forward = after_detconv

output,_ = dynm_model(image)

#----------------------------
# 原后处理
#----------------------------
output = non_max_suppression(output, 0.25, 0.65, nc=1, nkpt=17, kpt_label=True)
with torch.no_grad():
    output = output_to_target(output)
nimg = image[0].permute(1, 2, 0) * 255
nimg = nimg.cpu().numpy().astype(np.uint8)
nimg = cv2.cvtColor(nimg, 0)
for idx in range(output.shape[0]):
    plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

cv2.imshow(" ",nimg)
cv2.waitKey(0)
