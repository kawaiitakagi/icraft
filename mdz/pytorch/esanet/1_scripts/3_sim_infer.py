# -*- coding: utf-8 -*-
"""
lmy 2024-12-10 for esanet infer sunrgbd dataset
cd 1_script
run: python .\3_sim_infer.py --dataset sunrgbd
"""
from icraft.xir import *
from icraft.xrt import *
from icraft.host_backend import *
from icraft.buyibackend import *

from typing import List

import sys
sys.path.append(R"../0_esanet")
import argparse
from glob import glob
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.args import ArgumentParserRGBDSegmentation
from src.prepare_data import prepare_data

def run(network: Network, input: List[Tensor]) -> List[Tensor]:
    session = Session.Create([ HostBackend], network.view(0), [HostDevice.Default()])
    session.apply()
    output_tensors = session.forward( input ) #前向
    return output_tensors

GENERATED_JSON_FILE = "../3_deploy/modelzoo/esanet/imodel/8/esanet_BY.json"
GENERATED_RAW_FILE = "../3_deploy/modelzoo/esanet/imodel/8/esanet_BY.raw"


# 加载指令生成后的网络
generated_network = Network.CreateFromJsonFile(GENERATED_JSON_FILE)
generated_network.loadParamsFromFile(GENERATED_RAW_FILE)


def _load_img(fp):
    img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


if __name__ == "__main__":
    parser = ArgumentParserRGBDSegmentation(
        description='Efficient RGBD Indoor Sematic Segmentation (Inference)',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_common_args()
    parser.add_argument('--depth_scale', type=float,
                        default=1.0,
                        help='Additional depth scaling factor to apply.')
    args = parser.parse_args()

    # dataset
    args.pretrained_on_imagenet = False  # we are loading other weights anyway
    dataset, preprocessor = prepare_data(args, with_input_orig=True)
    n_classes = dataset.n_classes_without_void
    # pre process
    img_path=(R"../2_compile/qtset/sample/sample_rgb.png")
    depth_path = (R"../2_compile/qtset/sample/sample_depth.png")
    img_rgb = _load_img(img_path)
    img_depth = _load_img(depth_path).astype('float32')
    h, w, _ = img_rgb.shape

    # preprocess sample
    sample = preprocessor({'image': img_rgb, 'depth': img_depth})
    # add batch axis and copy to device
    image = sample['image'][None]
    depth = sample['depth'][None]

    input_0 =  np.transpose(np.array(image).astype(np.float32), (0, 2, 3, 1))
    input_1 =  np.transpose(np.array(depth).astype(np.float32), (0, 2, 3, 1))

    input_tensor_0 = Tensor(input_0, Layout("NHWC"))
    input_tensor_1 = Tensor(input_1, Layout("NHWC"))
    try:
        generated_output = run(generated_network, [input_tensor_0,input_tensor_1])
    except InternalError as i:
        print(i)

    print(np.array(generated_output[0]).shape)

    # post process
    pred = np.array(generated_output[0]).astype(np.float32)
    pred = np.transpose(pred, (0, 3, 1, 2))

    pred = torch.from_numpy(pred)
    pred = F.interpolate(pred, (h, w),
                        mode='bilinear', align_corners=False)
    pred = torch.argmax(pred, dim=1)
    pred = pred.cpu().numpy().squeeze().astype(np.uint8)

    # show result
    pred_colored = dataset.color_label(pred, with_void=False)
    fig, axs = plt.subplots(1, 3, figsize=(16, 3))
    [ax.set_axis_off() for ax in axs.ravel()]
    axs[0].imshow(img_rgb)
    axs[1].imshow(img_depth, cmap='gray')
    axs[2].imshow(pred_colored)
    plt.suptitle(f"Image depth "f"Model: icraft")
    plt.show()
