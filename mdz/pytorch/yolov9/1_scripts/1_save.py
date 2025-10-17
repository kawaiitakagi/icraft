import argparse
import torch
import torch.nn as nn 
import sys 
sys.path.append(R"../0_yolov9")

from models.yolo import BaseModel,DDetect
parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default= '../weights/yolov9-t-converted.pt', help='model path')
parser.add_argument('--imgsz', type=int, default= 640, help='img size')
parser.add_argument('--dst', type=str, default= '../2_compile/fmodel/yolov9t_640x640.pt', help='traced model path')
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

def new_DDetect_forward(self,x):
    "Only returns Conv results without other post process like make_anchors,dist2box  "
    shape = x[0].shape  # BCHW
    cls_dfl_head = []
    for i in range(self.nl):
        # cv3:80, means class probabilities
        # cv2:64, means predicted bounding boxes
        cls_dfl_head.append(self.cv3[i](x[i])) # cv3 对应80
        cls_dfl_head.append(self.cv2[i](x[i])) # cv2 对应64
    return cls_dfl_head
DDetect.forward = new_DDetect_forward

from models.common import DetectMultiBackend

# Load model
model = DetectMultiBackend(opt.weights)
# prepare dummy input
im = torch.ones((1,3,opt.imgsz,opt.imgsz))
pred = model(im, augment=False)

#----------------------------
# 添加隔离卷积（使用硬算子产生的限制）
#----------------------------
output_c = pred[1].size(1) #output_c = 64
fake_conv = nn.Conv2d(output_c, output_c, 1)
fake_conv.weight.data = torch.eye(output_c).view(output_c,output_c,1,1) #生成单位矩阵
fake_conv.bias.data = torch.zeros(output_c)
class New(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.conv = fake_conv #增加的伪卷积
        self.pre = model #先前的模型结构
    def forward(self,im):
        y = self.pre(im)
        for i in range(3):
            y[2 * i + 1] = self.conv(y[2 * i + 1]) #在channel = 64的分组卷积后 增加一层伪卷积
        return y 
new_model = New(model)
y = new_model(im)

new_y = torch.jit.trace(new_model,im,strict=False)
TRACE_PATH = opt.dst
torch.jit.save(new_y,TRACE_PATH)
print('TorchScript export success, saved in %s' % TRACE_PATH)
