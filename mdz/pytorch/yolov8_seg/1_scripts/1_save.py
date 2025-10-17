import sys
sys.path.append(R"../0_yolov8_seg")

from ultralytics import YOLO
from PIL import Image
from ultralytics.nn.modules.head import *
import torch

RED = '\033[31m'       # 设置前景色为红色
RESET = '\033[0m'      # 重置所有属性到默认值
ver = torch.__version__
assert ("1.6" in ver) or ("1.9" in ver), f"{RED}Unsupported PyTorch version: {ver}{RESET}"


# Load a model
weigths = torch.load("../weights/yolov8s-seg.pt" , map_location="cpu")
model = weigths['model'].float().eval()
model = model.fuse()

#  该脚本用于对导出trochscript模型，作为icraft的输入

#　对pose前向进行修改，只导出到卷积为止  源码：ultralytics.nn.modules.head.Segment

def seg_forward(self,x):
    p = self.proto(x[0])  # mask protos
    seg = []
    for i in range(self.nl):
        seg.append(self.cv4[i](x[i]))
    cls_dfl_head = []
    for i in range(self.nl):
        cls_dfl_head.append(self.cv3[i](x[i]))
        cls_dfl_head.append(self.cv2[i](x[i]))
    return cls_dfl_head[0],cls_dfl_head[1],seg[0],cls_dfl_head[2],cls_dfl_head[3],seg[1],cls_dfl_head[4],cls_dfl_head[5],seg[2],p

Segment.__call__ = seg_forward

# 导出torchscript模型用于icraft编译输入
# psin
torch.jit.save(torch.jit.trace(model,torch.randn(1,3,640,640,dtype=torch.float32)),"../2_compile/fmodel/yolov8s-seg-640x640.pt")
# plin
torch.jit.save(torch.jit.trace(model,torch.randn(1,3,512,960,dtype=torch.float32)),"../2_compile/fmodel/yolov8s-seg-512x960.pt")