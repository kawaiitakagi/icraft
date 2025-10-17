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
import sys
sys.path.append(R"../0_SFA3D-master")
sys.path.append(R"../0_SFA3D-master/sfa")
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


def process_result(result):
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
    detections = post_processing(detections, 3, 4, 0.2)
    # t2 = time_synchronized()

    detections = detections[0]  # only first batch
    return detections


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

GENERATED_JSON_FILE = "../2_compile/imodel/sfa3d/sfa3d_BY.json"
GENERATED_RAW_FILE = "../2_compile/imodel/sfa3d/sfa3d_BY.raw"

generated_network = Network.CreateFromJsonFile(GENERATED_JSON_FILE)
generated_network.loadParamsFromFile(GENERATED_RAW_FILE)


img_path = '../0_SFA3D-master/dataset/kitti/testing/image_2/000000.png'
calib_path = '../0_SFA3D-master/dataset/kitti/testing/calib/000000.txt'
lidar_path = '../0_SFA3D-master/dataset/kitti/testing/velodyne/000000.bin'

img, bev_map, calib = load_data(img_path, lidar_path, calib_path)
input_bev_maps = bev_map.float().cpu().numpy().transpose(0, 2, 3, 1)
# print(input_bev_maps.shape)
# input_bev_maps.tofile('input.bin')
# input_bev_maps = np.fromfile('input.bin', dtype=np.float32).reshape((1, 608, 608, 3))

generated_output = run(generated_network, [Tensor(input_bev_maps, Layout("NHWC"))])
result = []
for out in generated_output:
    result.append(np.array(out))
# with open('result.pkl', 'wb') as f:
#     pickle.dump(output, f)
# with open('result.pkl', 'rb') as f:
#     result = pickle.load(f)

detections = process_result(result)
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
cv2.imshow(" ", out_img)
cv2.waitKey(0)