from icraft.xir import *
from icraft.xrt import *
from icraft.host_backend import *
from icraft.buyibackend import *
import numpy as np
import pandas as pd
from numpy.linalg import norm
from typing import List
import platform
import cv2
import json
import os
from tqdm import tqdm
import pickle
import torch
import sys
sys.path.append(R"../0_SLBAF-Net-20.04/modules/yolov5-dual")
from utils.augmentations import letterbox
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)

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
    return img.numpy()


def make_grid(nx=20, ny=20, i=0):
    shape = 1, 3, ny, nx, 2  # grid shape
    y, x = torch.arange(ny), torch.arange(nx)

    yv, xv = torch.meshgrid(y, x)
    grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
    anchor_grid = anchors[i].view((1, 3, 1, 1, 2)).expand(shape)
    return grid, anchor_grid


def sim(network :Network, input: List[Tensor]) -> List[Tensor]:
    session = Session.Create([ HostBackend ], network.view(0), [ HostDevice.Default() ])
    session.enableTimeProfile(True)
    session.apply()
    output_tensors = session.forward( input )
    return output_tensors


def run(network: Network, input: List[Tensor]) -> List[Tensor]:
    URL_PATH=""
    if platform.machine() == 'aarch64':
        URL_PATH = R"axi://ql100aiu?npu=0x40000000&dma=0x80000000";
    else:
        URL_PATH = Rf"socket://ql100aiu@192.168.125.{ip_octet}:9981?npu=0x40000000&dma=0x80000000"
    # Buyi_device = Device.Open(URL_PATH)
    session = Session.Create([ HostBackend], network.view(0), [HostDevice.Default()])
    # session = Session.Create([ BuyiBackend, HostBackend], network.view(0), [ Buyi_device,HostDevice.Default()])
    session.enableTimeProfile(True) #打开计时功能
    session.apply()
    output_tensors = session.forward( input ) #前向

    # 计算时间
    result = session.timeProfileResults() #获取时间，[总时间，传输时间, 硬件时间，余下时间]
    # save_time(network, result, result_table_name)
    time = np.array(list(result.values()))
    total_softtime = np.sum(time[:,1])
    total_hardtime = np.sum(time[:,2])
    # print("total_softtime:{},total_hardtime:{}(ms)",total_softtime,total_hardtime)
    return output_tensors


ip_octet = 93
GENERATED_JSON_FILE = "../2_compile/imodel/SLBAF/SLBAF_BY.json"
GENERATED_RAW_FILE = "../2_compile/imodel/SLBAF/SLBAF_BY.raw"

generated_network = Network.CreateFromJsonFile(GENERATED_JSON_FILE)
generated_network.loadParamsFromFile(GENERATED_RAW_FILE)


imgsz = [640, 640]
stride = 32
names = ['pedestrian', 'cyclist', 'car', 'bus', 'truck', 'traffic_light', 'traffic_sign']
conf_thres, iou_thres, classes, agnostic_nms, max_det = 0.05, 0.45, None, False, 1000
anchors = torch.tensor([[[6,10], [16,30], [33,23]], [[30,61], [62,45], [59,119]], [[116,90], [156,198], [373,326]]])

im = load_image('../0_SLBAF-Net-20.04/modules/yolov5-dual/data/test_images/009000.jpg', imgsz, stride).transpose(0, 2, 3, 1)
im2 = load_image('../0_SLBAF-Net-20.04/modules/yolov5-dual/data/test_images2/009000.jpg', imgsz, stride).transpose(0, 2, 3, 1)
im0s = cv2.imread('../0_SLBAF-Net-20.04/modules/yolov5-dual/data/test_images/009000.jpg')

input1, input2 = Tensor(np.ascontiguousarray(im), Layout("NHWC")), Tensor(np.ascontiguousarray(im2), Layout("NHWC"))
# img1 = np.fromfile('009000.ftmp', dtype=np.float32).reshape((1, 480, 640, 3))
# img2 = np.fromfile('009000_r.ftmp', dtype=np.float32).reshape((1, 480, 640, 3))
# input1, input2 = Tensor(img1, Layout("NHWC")), Tensor(img2, Layout("NHWC"))
generated_output = run(generated_network, [input1, input2])

outputs = []
for out in generated_output:
    outputs.append(torch.from_numpy(np.array(out).transpose(0, 3, 1, 2)))


# print(outputs[0])
# with open('result.pkl', 'wb') as f:
#     pickle.dump(outputs, f)

conf_thres, iou_thres, classes, agnostic_nms, max_det = 0.05, 0.45, None, False, 1000
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
    det[:, :4] = scale_coords([480, 640], det[:, :4], im0s.shape).round()
    result_image = vis(im0s, boxes=det[:,:4], scores=det[:,4], cls_ids=det[:,5], conf=conf_thres, class_names=names)

cv2.imshow(" ", im0s)
cv2.waitKey(0)