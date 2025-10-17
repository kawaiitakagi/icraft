# -*- coding:utf-8 -*-
# 该脚本用来保存torchscript模型
import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import sys
sys.path.append(R"../0_SeResNet")
from senet.se_resnet import se_resnet34
from senet.se_module import SELayer

RED = '\033[31m'       # 设置前景色为红色
RESET = '\033[0m'      # 重置所有属性到默认值
ver = torch.__version__
assert ("1.6" in ver) or ("1.9" in ver), f"{RED}Unsupported PyTorch version: {ver}{RESET}"

def SELayer_forward(self, x):
    b, c, _, _ = x.size()
    y = self.avg_pool(x).view(b, c)
    y = self.fc(y).view(b, c, 1, 1)
    # return x * y.expand_as(x)
    return x*y 
SELayer.forward = SELayer_forward
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 文件及参数配置
    weights_path = "../weights/SeResNet34.pth"
    pt_path = '../2_compile/fmodel/SeResNet34_224x224.pt'
    # create model
    model = se_resnet34(num_classes=5).to(device)
    # load model weights
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    torch.jit.save(torch.jit.trace(model,torch.randn(1,3,224,224)),pt_path)
    print('TorchScript export success, saved as %s' % pt_path)
if __name__ == '__main__':
    main()