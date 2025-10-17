import argparse #命令行解析模块
import os
import platform
import sys
sys.path.append(R"../0_SLBAF-Net-20.04/modules/yolov5-dual")

import cv2
import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox
from models.yolo import Detect


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,])


def vis(img, boxes, scores, cls_ids, conf=0.2, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


def load_image(img_path, imgsz, stride, pt=True):
    img = cv2.imread(img_path)
    img = letterbox(img, imgsz, stride, pt)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float().unsqueeze(0)
    img /= 255
    return img


def new_forward(self, x):
    for i in range(self.nl):
        x[i] = self.m[i](x[i])  # conv
    return x


def make_grid(nx=20, ny=20, i=0):
    shape = 1, 3, ny, nx, 2  # grid shape
    y, x = torch.arange(ny), torch.arange(nx)

    yv, xv = torch.meshgrid(y, x)
    grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
    anchor_grid = anchors[i].view((1, 3, 1, 1, 2)).expand(shape)
    return grid, anchor_grid


anchors = torch.tensor([[[6,10], [16,30], [33,23]], [[30,61], [62,45], [59,119]], [[116,90], [156,198], [373,326]]])

weights = '../0_SLBAF-Net-20.04/modules/yolov5-dual/weights/dual.pt'
device = torch.device('cpu')
dnn = False
data = '../0_SLBAF-Net-20.04/modules/yolov5-dual/data/coco128.yaml'
half = False
model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)

imgsz = [640, 640]
stride, names, pt = model.stride, model.names, model.pt

Detect.forward = new_forward


im = load_image('../0_SLBAF-Net-20.04/modules/yolov5-dual/data/test_images/009000.jpg', imgsz, stride)
im2 = load_image('../0_SLBAF-Net-20.04/modules/yolov5-dual/data/test_images2/009000.jpg', imgsz, stride)
im0s = cv2.imread('../0_SLBAF-Net-20.04/modules/yolov5-dual/data/test_images/009000.jpg')
outputs = model(im, im2, augment=False, visualize=False, val=True)

conf_thres, iou_thres, classes, agnostic_nms, max_det = 0.25, 0.45, None, False, 1000
z = []
grid = [torch.zeros(1)] * 3
anchor_grid = [torch.zeros(1)] * 3
stride = [8, 16, 32]
for i in range(3):
    bs, _ , ny, nx = outputs[i].shape
    outputs[i] = outputs[i].view(bs, 3, 12, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

    grid[i], anchor_grid[i] = make_grid(nx, ny, i)
    y = outputs[i].sigmoid()
    y[..., 0:2] = (y[..., 0:2] * 2 + grid[i]) * stride[i]  # xy
    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
    z.append(y.view(bs, -1, 12))

pred = torch.cat(z, 1)  #torch.size([1, 25200, 85])
pred1 = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
det = pred1[0]
    # print(len(det))
if len(det):
    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()
    result_image = vis(im0s, boxes=det[:,:4], scores=det[:,4], cls_ids=det[:,5], conf=conf_thres, class_names=names)

cv2.imshow(" ", im0s)
cv2.waitKey(0)