import sys
sys.path.append(R"../../../Deps/modelzoo")
import icraft
from icraft.xir import *
from icraft.xrt import *
from icraft.buyibackend import *
from icraft.host_backend import *
from pyrtutils.icraft_utils import *
from pyrtutils.Netinfo import Netinfo
from pyrtutils.utils import *
from pyrtutils.et_device import *
from pyrtutils.calctime_utils import *

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
import yaml
sys.path.append(R"../../../../0_SLBAF-Net-20.04/modules/yolov5-dual")
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


def make_grid(anchors, nx=20, ny=20, i=0):
    anchors = torch.tensor(anchors)
    shape = 1, 3, ny, nx, 2  # grid shape
    y, x = torch.arange(ny), torch.arange(nx)

    yv, xv = torch.meshgrid(y, x)
    grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
    anchor_grid = anchors[i].view((1, 3, 1, 1, 2)).expand(shape)
    return grid, anchor_grid


if __name__ == "__main__":
    Yaml_Path = "../cfg/SLBAF-Net.yaml"
    if len(sys.argv) < 2:
        print("Info:未传入yaml参数,读入默认yaml文件: {}进行相关配置.".format(Yaml_Path))
    if len(sys.argv) == 2:
        Yaml_Path = sys.argv[1]
        print("info:传入yaml文件: {}进行相关配置.".format(Yaml_Path))
    if len(sys.argv) > 2:
        print("info:传入参数数量错误,请检查运行命令!")
        sys.exit(1)

    cfg = yaml.load(open(Yaml_Path, "r"), Loader=yaml.FullLoader)   
    folderPath = cfg["imodel"]["dir"]
    stage = cfg["imodel"]["stage"]
    run_sim = cfg["imodel"]["sim"]
    JSON_PATH, RAW_PATH = getJrPath(folderPath, stage, run_sim)

    load_mmu = cfg["imodel"]["mmu"]
    load_speedmode = cfg["imodel"]["speedmode"]
    load_compressFtmp = cfg["imodel"]["compressFtmp"]
    ip = str(cfg["imodel"]["ip"])
    save = cfg["imodel"]["save"]
    show = cfg["imodel"]["show"]

    imgRoot1 = os.path.abspath(cfg["dataset"]["dir"])
    imgList = os.path.abspath(cfg["dataset"]["list"])
    imgRoot2 = os.path.abspath(cfg["dataset"]["dir2"])
    names_path = cfg["dataset"]["names"]
    resRoot = cfg["dataset"]["res"]
    if not os.path.exists(resRoot):
        os.mkdir(resRoot)
    # 模型自身相关参数配置
    conf = cfg["param"]["conf"]
    iou_thresh = cfg["param"]["iou_thresh"]
    multilabel = cfg["param"]["multilabel"]
    number_of_class = cfg["param"]["number_of_class"]
    anchors = cfg["param"]["anchors"]
    fpga_nms = cfg["param"]["fpga_nms"]
    imgsz = [640, 640]
    stride = 32
    strides = [8, 16, 32]
    names = ['pedestrian', 'cyclist', 'car', 'bus', 'truck', 'traffic_light', 'traffic_sign']

    # 加载network
    network = loadNetwork(JSON_PATH, RAW_PATH)
    # 初始化netinfo
    netinfo = Netinfo(network)
    # 选择对网络进行切分
    network_view = network.view(netinfo.inp_shape_opid + 1)
    # 打开device
    device = openDevice(run_sim, ip, netinfo.mmu or load_mmu)
    # 初始化session
    session = initSession(run_sim, network_view, device, netinfo.mmu or load_mmu, load_speedmode, load_compressFtmp)
    # 开启计时功能
    session.enableTimeProfile(True)
    # session执行前必须进行apply部署操作
    session.apply()

    for line in tqdm(open(imgList, "r")):
        line = line.strip()
        img_path = os.path.join(imgRoot1, line)
        dep_path = os.path.join(imgRoot2, line)
        im = load_image(img_path, imgsz, stride).transpose(0, 2, 3, 1)
        im2 = load_image(dep_path, imgsz, stride).transpose(0, 2, 3, 1)
        im_ori = cv2.imread(img_path)
        input1, input2 = Tensor(np.ascontiguousarray(im), Layout("NHWC")), Tensor(np.ascontiguousarray(im2), Layout("NHWC"))

        output_tensors = session.forward([input1, input2])

        if not run_sim: 
            device.reset(1)
            # calctime_detail(session)
            calctime_detail(session,network, name="./"+network.name+"_time.xlsx")

        outputs = []
        for out in output_tensors:
            outputs.append(torch.from_numpy(np.array(out).transpose(0, 3, 1, 2)))

        z = []
        grid = [torch.zeros(1)] * 3
        anchor_grid = [torch.zeros(1)] * 3
        for i in range(3):
            bs, _ , ny, nx = outputs[i].shape
            outputs[i] = outputs[i].view(bs, 3, 12, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            grid[i], anchor_grid[i] = make_grid(anchors, nx, ny, i)
            y = outputs[i].sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2 + grid[i]) * strides[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
            z.append(y.view(bs, -1, 12))

        pred = torch.cat(z, 1)  #torch.size([1, 25200, 85])
        pred1 = non_max_suppression(pred, conf, iou_thresh, None, False, max_det=1000)
        det = pred1[0]
        # print(len(det))
        if len(det):
            det[:, :4] = scale_coords([480, 640], det[:, :4], im_ori.shape).round()
            result_image = vis(im_ori, boxes=det[:,:4], scores=det[:,4], cls_ids=det[:,5], conf=conf, class_names=names)

        cv2.imwrite(resRoot+'/'+line, result_image)


    if not run_sim:
        Device.Close(device)
