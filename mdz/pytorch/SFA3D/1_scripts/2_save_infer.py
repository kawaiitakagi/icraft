import sys
sys.path.append(R"../0_SFA3D-master")
sys.path.append(R"../0_SFA3D-master/sfa")
import argparse
import os
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from easydict import EasyDict as edict
import cv2
import torch
import numpy as np
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


img_path = '../0_SFA3D-master/dataset/kitti/testing/image_2/000000.png'
calib_path = '../0_SFA3D-master/dataset/kitti/testing/calib/000000.txt'
lidar_path = '../0_SFA3D-master/dataset/kitti/testing/velodyne/000000.bin'

img, bev_map, calib = load_data(img_path, lidar_path, calib_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = torch.jit.load('./sfa3d.pt')
net.to(device)
net.eval()
input_bev_maps = bev_map.to(device).float()

with torch.no_grad():
    outputs = net(input_bev_maps)

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


detections = detections[0]  # only first batch

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
# cv2.imwrite('test.png', out_img)
cv2.imshow(" ", out_img)
cv2.waitKey(0)