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
import torch
from torch.nn import functional as F
import yaml
import sys


def Normalization(image):
    in_ = image[:, :, ::-1]
    in_ -= np.array((123.675, 116.28, 103.53))
    in_ /= np.array((58.395, 57.12, 57.375))
    return in_


def load_image(im, image_size=320):
    in_ = np.array(im, dtype=np.float32)
    in_ = cv2.resize(in_, (image_size, image_size))
    in_ = Normalization(in_)
    in_ = in_.transpose((2, 0, 1))
    return in_


if __name__ == "__main__":
    Yaml_Path = "../cfg/JL-DCF.yaml"
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

    imgRoot1 = os.path.abspath(cfg["dataset"]["dir1"])
    imgList1 = os.path.abspath(cfg["dataset"]["list1"])
    imgRoot2 = os.path.abspath(cfg["dataset"]["dir2"])
    imgList2 = os.path.abspath(cfg["dataset"]["list2"])
    names_path = cfg["dataset"]["names"]
    resRoot = cfg["dataset"]["res"]
    if not os.path.exists(resRoot):
        os.mkdir(resRoot)
    # 模型自身相关参数配置
    # conf = cfg["param"]["conf"]
    # iou_thresh = cfg["param"]["iou_thresh"]
    # multilabel = cfg["param"]["multilabel"]
    # number_of_class = cfg["param"]["number_of_class"]
    # anchors = cfg["param"]["anchors"]
    # fpga_nms = cfg["param"]["fpga_nms"]

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

    rgb_list = [line.strip() for line in open(imgList1, "r")]
    dep_list = [line.strip() for line in open(imgList2, "r")]
    for idx in range(len(rgb_list)):
        rgb_path = os.path.join(imgRoot1, rgb_list[idx])
        dep_path = os.path.join(imgRoot2, dep_list[idx])
        images = cv2.imread(rgb_path)
        shape = images.shape[:2]
        depth = cv2.imread(dep_path)
        images = np.ascontiguousarray(np.expand_dims(load_image(images).transpose(1, 2, 0), axis=0))
        depth = np.ascontiguousarray(np.expand_dims(load_image(depth).transpose(1, 2, 0), axis=0))
        input1, input2 = Tensor(images, Layout("NHWC")), Tensor(depth, Layout("NHWC"))

        output_tensors = session.forward([input1, input2])

        if not run_sim: 
            device.reset(1)
            # calctime_detail(session)
            calctime_detail(session,network, name="./"+network.name+"_time.xlsx")

        output = np.array(output_tensors[0]).transpose(0, 3, 1, 2)
        preds = torch.from_numpy(output)

        preds = F.interpolate(preds, shape, mode='bilinear', align_corners=True)
        pred = np.squeeze(torch.sigmoid(preds)).cpu().data.numpy()

        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        multi_fuse = 255 * pred
        cv2.imwrite(resRoot+'/'+rgb_list[idx], multi_fuse)

    if not run_sim:
        Device.Close(device)