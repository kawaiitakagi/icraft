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

# src_dir = os.path.dirname("../0_SFA3D-master")
# while not src_dir.endswith("sfa"):
#     src_dir = os.path.dirname(src_dir)
# if src_dir not in sys.path:
#     sys.path.append(src_dir)

from data_process.kitti_dataloader import create_test_dataloader
from models.model_utils import create_model
from utils.misc import make_folder, time_synchronized
from utils.evaluation_utils import decode, post_processing, draw_predictions, convert_det_to_real_values
from utils.torch_utils import _sigmoid
import config.kitti_config as cnf
from data_process.transformation import lidar_to_camera_box
from utils.visualization_utils import merge_rgb_to_bev, show_rgb_image_with_boxes
from data_process.kitti_data_utils import Calibration
import torch.nn.functional as F


class cfg:
    pass


configs = cfg()
configs.saved_fn = 'fpn_resnet_18'
configs.arch = 'fpn_resnet_18'
configs.pretrained_path = '../0_SFA3D-master/checkpoints/fpn_resnet_18/fpn_resnet_18_epoch_300.pth'
configs.K = 50
configs.no_cuda = False
configs.gpu_idx = 0
configs.num_samples = None
configs.num_workers = 0
configs.batch_size = 1
configs.peak_thresh = 0.2
configs.save_test_output = True
configs.output_format = 'image'
configs.output_video_fn = None
configs.output_width = 608
configs.pin_memory = True
configs.distributed = False  # For testing on 1 GPU only

configs.input_size = (608, 608)
configs.hm_size = (152, 152)
configs.down_ratio = 4
configs.max_objects = 50

configs.imagenet_pretrained = False
configs.head_conv = 64
configs.num_classes = 3
configs.num_center_offset = 2
configs.num_z = 1
configs.num_dim = 3
configs.num_direction = 2  # sin, cos

configs.heads = {
    'hm_cen': configs.num_classes,
    'cen_offset': configs.num_center_offset,
    'direction': configs.num_direction,
    'z_coor': configs.num_z,
    'dim': configs.num_dim
}
configs.num_input_features = 4

####################################################################
##############Dataset, Checkpoints, and results dir configs#########
####################################################################
configs.root_dir = '../0_SFA3D-master'
configs.dataset_dir = os.path.join(configs.root_dir, 'dataset', 'kitti')

# model = create_model(configs)
# print('\n\n' + '-*=' * 30 + '\n\n')
# assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
# model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
# print('Loaded weights from {}\n'.format(configs.pretrained_path))

# configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
# model = model.to(device=configs.device)

# out_cap = None

# model.eval()

test_dataloader = create_test_dataloader(configs)

with torch.no_grad():
    for batch_idx, batch_data in enumerate(test_dataloader):
        metadatas, bev_maps, img_rgbs = batch_data
        input_idx = int(metadatas['img_path'][0].split('\\')[-1].split('.')[0])
        input_bev_maps = bev_maps.float()
        dump_input = input_bev_maps.cpu().numpy().transpose(0, 2, 3, 1)
        dump_input.tofile('./qtset/{:06d}.ftmp'.format(input_idx))

        if batch_idx >= 15:
            break
