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
import yaml
import os
from tqdm import tqdm
import pickle
import sys
sys.path.append(R"../../../../0_SFA3D-master")
sys.path.append(R"../../../../0_SFA3D-master/sfa")
import argparse
import os
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from easydict import EasyDict as edict
import torch
from data_process.kitti_data_utils import gen_hm_radius, compute_radius, Calibration, get_filtered_lidar
from data_process.kitti_bev_utils import makeBEVMap, drawRotatedBox, get_corners
from data_process import transformation
import config.kitti_config as cnf
from utils.torch_utils import _sigmoid
import torch.nn.functional as F
from utils.evaluation_utils import decode, post_processing, draw_predictions, convert_det_to_real_values
from data_process.transformation import lidar_to_camera_box
from utils.visualization_utils import merge_rgb_to_bev, show_rgb_image_with_boxes


def load_data(img_path, lidar_path, calib_path):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    lidarData = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    calib = Calibration(calib_path)
    lidarData = get_filtered_lidar(lidarData, cnf.boundary)
    bev_map = makeBEVMap(lidarData, cnf.boundary)
    bev_map = torch.from_numpy(bev_map).unsqueeze(0)

    return img, bev_map, calib


def process_result(result, number_of_class, conf):
    outputs = []
    for out in result:
        outputs.append(torch.from_numpy(out))

    for idx, outs in enumerate(outputs):
        softmax_outs = F.softmax(outs, dim=-1)
        ret_outs = (outs * softmax_outs).sum(dim=-1)
        outputs[idx] = ret_outs
    outputs[0] = _sigmoid(outputs[0])
    outputs[1] = _sigmoid(outputs[1])
    detections = decode(outputs[0], outputs[1], outputs[2], outputs[3],
                    outputs[4], K=50)

    detections = detections.cpu().numpy().astype(np.float32)
    detections = post_processing(detections, number_of_class, 4, conf)
    # t2 = time_synchronized()

    detections = detections[0]  # only first batch
    return detections


if __name__ == "__main__":
    Yaml_Path = "../cfg/SFA3D.yaml"
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

    imgRoot = os.path.abspath(cfg["dataset"]["dir1"])
    pcRoot = os.path.abspath(cfg["dataset"]["dir2"])
    calibRoot = os.path.abspath(cfg["dataset"]["dir3"])
    imgList = os.path.abspath(cfg["dataset"]["list"])

    names_path = cfg["dataset"]["names"]
    resRoot = cfg["dataset"]["res"]
    if not os.path.exists(resRoot):
        os.mkdir(resRoot)
    # 模型自身相关参数配置
    conf = cfg["param"]["conf"]
    multilabel = cfg["param"]["multilabel"]
    number_of_class = cfg["param"]["number_of_class"]

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
        img_path = os.path.join(imgRoot, line)
        pc_path = os.path.join(pcRoot, line.replace('png', 'bin'))
        calib_path = os.path.join(calibRoot, line.replace('png', 'txt'))

        img, bev_map, calib = load_data(img_path, pc_path, calib_path)
        input_bev_maps = np.ascontiguousarray(bev_map.float().cpu().numpy().transpose(0, 2, 3, 1))

        input = Tensor(input_bev_maps, Layout("NHWC"))
        # dmaInit(run_sim,netinfo.ImageMake_on, netinfo.i_shape[0][1:],input_tensor, device)
        # run
        output_tensors = session.forward([input])

        if not run_sim: 
            device.reset(1)
            # calctime_detail(session)
            calctime_detail(session,network, name="./"+network.name+"_time.xlsx")

        result = []
        for out in output_tensors:
            result.append(np.array(out))

        detections = process_result(result, number_of_class, conf)
        bev_map = (bev_map.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        bev_map = cv2.resize(bev_map, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
        bev_map = draw_predictions(bev_map, detections.copy(), 3)

        # Rotate the bev_map
        bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)
        kitti_dets = convert_det_to_real_values(detections)
        if len(kitti_dets) > 0:
            kitti_dets[:, 1:] = lidar_to_camera_box(kitti_dets[:, 1:], calib.V2C, calib.R0, calib.P2)
            img = show_rgb_image_with_boxes(img, kitti_dets, calib)

        out_img = merge_rgb_to_bev(img, bev_map, output_width=608)
        cv2.imwrite(resRoot+'/'+line, out_img)

    if not run_sim:
        Device.Close(device)