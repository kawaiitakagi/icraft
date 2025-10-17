# 该脚本用来加载模型并导出为torchscript模型（去除部分后处理）
import sys
sys.path.append(R"../0_yolov5_7.0")
import argparse
import os
import cv2
import torch
import numpy as np
from utils.augmentations import letterbox
from models.experimental import attempt_load
from models.yolo import Detect


# 修改前向，去除前向中后处理操作
def new_forward(self, x):
    for i in range(self.nl):
        x[i] = self.m[i](x[i])  # conv
    return x 


def save_trace_model(img_path, model_path, test_size):

    h, w = opt.imgsz[0],opt.imgsz[1]
    dumyin = torch.randn(1,3,h, w,dtype=torch.float32)

    # 加载模型
    model = attempt_load(model_path)
    Detect.forward = new_forward  # replace the forward of Detect 
    # 保存 torchscript model
    y = torch.jit.trace(model, dumyin ,strict=False) 
    torch.jit.save(y, TRACE_PATH)
    print('TorchScript export success, saved in %s' % TRACE_PATH)
    # plin
    torch.jit.save(torch.jit.trace(model,torch.randn(1,3,352,640,dtype=torch.float32)),"../2_compile/fmodel/yolov5s_352x640.pt")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_pt', type=str, default=R'../weights/yolov5s.pt', help='model path')
    parser.add_argument('--trace_path', type=str, default=R"../2_compile/fmodel/yolov5s_640x640.pt", help='trace path')
    parser.add_argument('--source', type=str, default=R'../0_yolov5_7.0/data/images/bus.jpg', help='image path')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640], help='image size')
    opt = parser.parse_args()

    TRACE_PATH = opt.trace_path
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    test_size = tuple(opt.imgsz)

    if os.path.isfile(opt.source):
        save_trace_model(opt.source, opt.model_pt, test_size)
    elif os.path.isdir(opt.source):
        image_list = os.listdir(opt.source)
        for image_file in image_list:
            image_path = opt.source + "//" + image_file 
            save_trace_model(image_path, opt.model_pt, test_size)
    
    