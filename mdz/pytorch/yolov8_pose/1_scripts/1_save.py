import sys
sys.path.append(R"../0_yolov8_pose")

from ultralytics import YOLO
from PIL import Image
from ultralytics.nn.modules.head import *
import torch

RED = '\033[31m'       # 设置前景色为红色
RESET = '\033[0m'      # 重置所有属性到默认值
ver = torch.__version__
assert ("1.6" in ver) or ("1.9" in ver), f"{RED}Unsupported PyTorch version: {ver}{RESET}"


# Load a model
weigths = torch.load("../weights/yolov8s-pose.pt" , map_location="cpu")
model = weigths['model'].float().eval()
model = model.fuse()

#  该脚本用于对导出trochscript模型，作为icraft的输入

#　对pose前向进行修改，只导出到卷积为止  源码：ultralytics.nn.modules.head.Pose

def pose_forward(self,x):
    kpt = []
    for i in range(self.nl):
        kpt.append(self.cv4[i](x[i]))
    cls_dfl_head = []
    for i in range(self.nl):
        cls_dfl_head.append(self.cv3[i](x[i]))
        cls_dfl_head.append(self.cv2[i](x[i]))
    return cls_dfl_head[0],cls_dfl_head[1],kpt[0],cls_dfl_head[2],cls_dfl_head[3],kpt[1],cls_dfl_head[4],cls_dfl_head[5],kpt[2]

Pose.__call__ = pose_forward

# 导出torchscript模型用于icraft编译输入

torch.jit.save(torch.jit.trace(model,torch.randn(1,3,640,640,dtype=torch.float32)),"../2_compile/fmodel/yolov8s-pose-640x640.pt")
torch.jit.save(torch.jit.trace(model,torch.randn(1,3,352,640,dtype=torch.float32)),"../2_compile/fmodel/yolov8s-pose-352x640.pt")