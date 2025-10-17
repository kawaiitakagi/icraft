# -*- coding:utf-8 -*-
# 该脚本用来保存torchscript模型
import argparse
import sys
import os
import time

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
sys.path.append(R"../0_yolov7")
import torch
import torch.nn as nn

import models
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size
from utils.torch_utils import select_device
from models.common import *
from models.yolo import *

RED = '\033[31m'       # 设置前景色为红色
RESET = '\033[0m'      # 重置所有属性到默认值
ver = torch.__version__
assert ("1.6" in ver) or ("1.9" in ver), f"{RED}Unsupported PyTorch version: {ver}{RESET}"

#----------------------------
# 等效reorg
#----------------------------
REORG_WEIGHTS=torch.zeros(12,3,2,2)
for i in range(12):
    j=i%3
    k=(i//3)%2
    l=0 if i<6 else 1
    REORG_WEIGHTS[i,j,k,l]=1

REORG_BIAS = torch.zeros(12)

def reorg_forward(self,x):
    y = F.conv2d(x,REORG_WEIGHTS,bias=REORG_BIAS,stride=2,padding=0,dilation=1,groups=1)
    return y

ReOrg.__call__ = reorg_forward


def detect_forward(self, x):
    z = []  # inference output
    self.training |= self.export
    for i in range(self.nl):
        x[i] = self.m[i](x[i])  # conv
        bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        # x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous() # 模型导出时 需要将此行注释掉
        if not self.training:  # inference
            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
            y = x[i].sigmoid()
            if not torch.onnx.is_in_onnx_export():
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            else:
                xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                y = torch.cat((xy, wh, conf), 4)
            z.append(y.view(bs, -1, self.no))

    if self.training:
        out = x
    elif self.end2end:
        out = torch.cat(z, 1)
    elif self.include_nms:
        z = self.convert(z)
        out = (z, )
    elif self.concat:
        out = torch.cat(z, 1)
    else:
        out = (torch.cat(z, 1), x)
    return out

Detect.forward = detect_forward
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='..\weights\yolov7.pt', help='weights path')
    parser.add_argument('--img_size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--export_dir', type=str, default=R'..\2_compile\fmodel', help='weights path')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')
    parser.add_argument('--grid', action='store_true', help='export Detect() layer grid')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    
    #print(opt)
    set_logging()
    t = time.time()

    # Load PyTorch model
    device = select_device(opt.device)
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    labels = model.names

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size).to(device)  # image size(1,3,320,192) iDetection

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.9.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                pass

    model.model[-1].export = not opt.grid  # set Detect() layer grid export
    y = model(img)  # dry run

    # TorchScript export
    try:
        print('\nStarting TorchScript export with torch %s...' % torch.__version__)
        if not os.path.exists(opt.export_dir):
            os.makedirs(opt.export_dir)

        f = os.path.join(opt.export_dir,  opt.weights.split('\\')[-1].split('.')[0] + f'_{str(opt.img_size[0])}x{str(opt.img_size[1])}.pt')  # filename
        ts = torch.jit.trace(model, torch.rand(1, 3, opt.img_size[0], opt.img_size[1], dtype=torch.float), strict=False)  #  img -> torch.rand(1,3,1280,1280,dtype=torch.float) 
        ts.save(f)
        print('TorchScript export success, saved as %s' % f)
    except Exception as e:
        print('TorchScript export failure: %s' % e)
