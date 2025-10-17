# -*- coding: utf-8 -*-
"""
lmy 2024-12-10 for esanet infer sunrgbd dataset
cd 1_script
run: python .\2_save_infer.py --dataset sunrgbd
"""

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
from src.build_model import build_model
from src.prepare_data import prepare_data

def _load_img(fp):
    img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

if __name__ == '__main__':
    # arguments
    parser = ArgumentParserRGBDSegmentation(
        description='Efficient RGBD Indoor Sematic Segmentation (Inference)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_common_args()
    parser.add_argument('--depth_scale', type=float,
                        default=1.0,
                        help='Additional depth scaling factor to apply.')
    args = parser.parse_args()

    # dataset
    args.pretrained_on_imagenet = False  # we are loading other weights anyway
    dataset, preprocessor = prepare_data(args, with_input_orig=True)
    n_classes = dataset.n_classes_without_void

    # load sample
    img_rgb = _load_img('../2_compile/qtset/sample/sample_rgb.png')
    img_depth = _load_img('../2_compile/qtset/sample/sample_depth.png').astype('float32') * args.depth_scale
    h, w, _ = img_rgb.shape
    
    # preprocess sample
    sample = preprocessor({'image': img_rgb, 'depth': img_depth})

    # add batch axis and copy to device
    image = sample['image'][None]
    depth = sample['depth'][None]

    loaded_model = torch.jit.load("../2_compile/fmodel/esanet_480x640_sunrgbd.pt")
    loaded_model.eval()
    # apply network
    pred = loaded_model(image, depth)

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
    plt.suptitle(f"Image depth" f"Model:pt")
    plt.show()

