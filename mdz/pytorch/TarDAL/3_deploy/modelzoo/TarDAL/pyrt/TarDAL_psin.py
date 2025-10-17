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

import sys
import numpy as np
import pandas as pd
from numpy.linalg import norm
from typing import List
import platform
import cv2
import json
import os
from tqdm import tqdm
from kornia import tensor_to_image, image_to_tensor
from kornia.color import ycbcr_to_rgb, rgb_to_bgr, rgb_to_ycbcr, bgr_to_rgb
import torch
from torchvision.transforms import Resize
import yaml


def gray_read(img_path):
    img_n = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    img_t = image_to_tensor(img_n).float() / 255
    return img_t


def ycbcr_read(img_path):
    img_n = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    img_t = image_to_tensor(img_n).float() / 255
    img_t = rgb_to_ycbcr(bgr_to_rgb(img_t))
    y, cbcr = torch.split(img_t, [1, 2], dim=0)
    return y, cbcr


if __name__ == "__main__":
    Yaml_Path = "../cfg/TarDAL.yaml"
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
    imgList = os.path.abspath(cfg["dataset"]["list"])
    imgRoot2 = os.path.abspath(cfg["dataset"]["dir2"])
    # names_path = cfg["dataset"]["names"]
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

    for line in tqdm(open(imgList, "r")):
        line = line.strip()
        vi_path = os.path.join(imgRoot1, line)
        ir_path = os.path.join(imgRoot2, line)
        ori_img = cv2.imread(vi_path)
        vi, cbcr = ycbcr_read(vi_path)
        ir = gray_read(ir_path)
        max_size = torch.Size([768, 1024])
        transform_fn = Resize(size=max_size)
        t = torch.cat([ir, vi, cbcr], dim=0)
        ir, vi, cbcr = torch.split(transform_fn(t), [1, 1, 2], dim=0)
        ir = ir.unsqueeze(-1).numpy().transpose(3, 1, 2, 0)
        vi = vi.unsqueeze(-1).numpy().transpose(3, 1, 2, 0)
        cbcr = cbcr.unsqueeze(-1).numpy().transpose(3, 1, 2, 0)
        input1, input2 = Tensor(ir, Layout("NHWC")), Tensor(vi, Layout("NHWC"))
        # dmaInit(run_sim,netinfo.ImageMake_on, netinfo.i_shape[0][1:],input_tensor, device)
        # run
        output_tensors = session.forward([input1, input2])

        if not run_sim: 
            device.reset(1)
            calctime_detail(session,network, name="./"+network.name+"_time.xlsx")

        output = output_tensors[0]

        fuse = np.concatenate((output, cbcr), axis=3).transpose(0, 3, 1, 2)
        img = ycbcr_to_rgb(torch.from_numpy(fuse))
        img = rgb_to_bgr(img)
        img_n = tensor_to_image(img.squeeze().cpu()) * 255
        if ori_img.shape != img_n.shape:
            img_n = cv2.resize(img_n, (ori_img.shape[1], ori_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(resRoot+'/'+line, img_n)

    if not run_sim:
        Device.Close(device)