#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import time
import sys
import os
import torch
import torch.nn as nn
import onnx

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
sys.path.append(R"../0_yolov6")
from yolov6.models.yolo import *
from yolov6.models.effidehead import Detect
from yolov6.layers.common import *
from yolov6.utils.events import LOGGER
from yolov6.utils.checkpoint import load_checkpoint
from io import BytesIO
from yolov6.models.effidehead import Detect
from yolov6.models.yolo import Model

RED = '\033[31m'       # 设置前景色为红色
RESET = '\033[0m'      # 重置所有属性到默认值
ver = torch.__version__
assert ("1.6" in ver) or ("1.9" in ver), f"{RED}Unsupported PyTorch version: {ver}{RESET}"

def detect_forward(self, x):
    cls_score_list = []
    reg_dist_list = []

    for i in range(self.nl):
        b, _, h, w = x[i].shape
        l = h * w

        x[i] = self.stems[i](x[i])
        cls_x = x[i]
        reg_x = x[i]
        cls_feat = self.cls_convs[i](cls_x)
        cls_output = self.cls_preds[i](cls_feat)
        reg_feat = self.reg_convs[i](reg_x)
        reg_output = self.reg_preds[i](reg_feat)

        
        cls_score_list.append(cls_output)
        reg_dist_list.append(reg_output)
    return cls_score_list[0],reg_dist_list[0],cls_score_list[1],reg_dist_list[1],cls_score_list[2],reg_dist_list[2]
Detect.forward = detect_forward        

def model_forward(self, x):
    export_mode = torch.onnx.is_in_onnx_export()
    x = self.backbone(x)
    x = self.neck(x)
    if export_mode == False:
        featmaps = []
        featmaps.extend(x)
    x = self.detect(x)
    return x 
Model.forward = model_forward

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='../weights/yolov6s_v2_reopt.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help='set Detect() inplace=True')
    parser.add_argument('--simplify', action='store_true', help='simplify onnx model')
    parser.add_argument('--end2end', action='store_true', help='export end2end onnx')
    parser.add_argument('--trt-version', type=int, default=8, help='tensorrt version')
    parser.add_argument('--with-preprocess', action='store_true', help='export bgr2rgb and normalize')
    parser.add_argument('--max-wh', type=int, default=None, help='None for tensorrt nms, int value for onnx-runtime nms')
    parser.add_argument('--topk-all', type=int, default=100, help='topk objects for every images')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='iou threshold for NMS')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='conf threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()
    args.img_size *= 2 if len(args.img_size) == 1 else 1  # expand
    print(args)
    t = time.time()

    # Check device
    cuda = args.device != 'cpu' and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.device}' if cuda else 'cpu')
    assert not (device.type == 'cpu' and args.half), '--half only compatible with GPU export, i.e. use --device 0'
    # Load PyTorch model
    model = load_checkpoint(args.weights, map_location=device, inplace=True, fuse=True)  # load FP32 model
    for layer in model.modules():
        if isinstance(layer, RepVGGBlock):
            layer.switch_to_deploy()

    # Input
    img = torch.zeros(args.batch_size, 3, *args.img_size).to(device)  # image size(1,3,320,192) iDetection

    # Update model

    if args.half:
        img, model = img.half(), model.half()  # to FP16
    for k, m in model.named_modules():
        if isinstance(m, Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.SiLU):
                #m.act = SiLU()
                pass
        elif isinstance(m, Detect):
            m.trace = True  # 保存适配Icraft的模型
            m.inplace = args.inplace
    if args.end2end:
        from yolov6.models.end2end import End2End
        model = End2End(model, max_obj=args.topk_all, iou_thres=args.iou_thres,score_thres=args.conf_thres,
                        max_wh=args.max_wh, device=device, trt_version=args.trt_version, with_preprocess=args.with_preprocess)

    y = model(img)  # dry run
    # TorchScript export
    print('\nStarting TorchScript export with torch ')
    h, w = args.img_size
    dummyin = torch.randn(1,3,h,w,dtype = torch.float32)
    ts = torch.jit.trace(model, dummyin, strict=False)
    pb_path = os.path.join('../2_compile/fmodel/'+args.weights.split('/')[-1].split('.')[0] + f'_{str(args.img_size[0])}x{str(args.img_size[1])}.pt')
    ts.save(pb_path)
    print('TorchScript export success, save at: ',pb_path )