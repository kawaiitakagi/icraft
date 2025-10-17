import argparse
import torch
import torch.nn as nn 
import sys 
sys.path.append(R"../0_yolov9")

from models.yolo import BaseModel,Panoptic
from models.common import DetectMultiBackend
parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default= '../weights/gelan-c-pan.pt', help='model path')
parser.add_argument('--imgsz', type=int, default= 640, help='img size')
parser.add_argument('--dst', type=str, default= '../2_compile/fmodel/gelan-c-pan-640x640.pt', help='traced model path')
opt = parser.parse_args()

def new_predict_once(self, x, profile=False, visualize=False):
    y, dt = [], []  # outputs
    for m in self.model:
        if m.f != -1:  # if not from previous layer
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
        x = m(x)  # run
        y.append(x if m.i in self.save else None)  # save output
    return x
BaseModel._forward_once = new_predict_once

def new_Panoptic_forward(self,x):
    # just export conv   EX: models/yolo/Panoptic
    p = self.proto(x[0]) # 256: protos 
    s = self.uconv(x[0]) # 173: class 80 + semanic class 93 
    seg = []
    for i in range(self.nl):
        seg.append(self.cv4[i](x[i]))  # cv4: 32 mask coefficients
    cls_dfl_head = []
    for i in range(self.nl):
        # cv3:80, means class probabilities
        # cv2:64, means predicted bounding boxes
        cls_dfl_head.append(self.cv3[i](x[i])) 
        cls_dfl_head.append(self.cv2[i](x[i])) 
    return cls_dfl_head[0],cls_dfl_head[1],seg[0],cls_dfl_head[2],cls_dfl_head[3],seg[1],cls_dfl_head[4],cls_dfl_head[5],seg[2],p,s
Panoptic.forward = new_Panoptic_forward

# Load model
model = DetectMultiBackend(opt.weights)
model.eval()
# prepare dummy input
im = torch.ones((1,3,opt.imgsz,opt.imgsz))

y = torch.jit.trace(model,im,strict=False)
TRACE_PATH = opt.dst
torch.jit.save(y,TRACE_PATH)
print('TorchScript export success, saved in %s' % TRACE_PATH)