import numpy as np
import pandas as pd
from numpy.linalg import norm
from typing import List
import platform
import cv2
import json
import os
from tqdm import tqdm


def Normalization(image):
    in_ = image[:, :, ::-1]
    # in_ = in_ / 255.0
    # in_ -= np.array((0.485, 0.456, 0.406))
    # in_ /= np.array((0.229, 0.224, 0.225))
    in_ -= np.array((123.675, 116.28, 103.53))
    in_ /= np.array((58.395, 57.12, 57.375))
    return in_


def preprocess(im, image_size=(320, 320)):
    in_ = np.array(im, dtype=np.float32)
    in_ = cv2.resize(in_, image_size)
    in_ = Normalization(in_)
    return in_


img_dir = '../0_JL-DCF-pytorch-master/dataset/test/LFSD/RGB/'
dep_dir = '../0_JL-DCF-pytorch-master/dataset/test/LFSD/depth/'
img_list = sorted(os.listdir(img_dir))
dep_list = sorted(os.listdir(dep_dir))
for idx in range(len(img_list)):
    print(idx)
    img_path = img_dir + img_list[idx]
    dep_path = dep_dir + dep_list[idx]
    print(img_list[idx], dep_list[idx])
    images = cv2.imread(img_path)
    depth = cv2.imread(dep_path)
    images = np.ascontiguousarray(preprocess(images).reshape(1, 320, 320, 3))
    depth = np.ascontiguousarray(preprocess(depth).reshape(1, 320, 320, 3))
    images.tofile('./icraft/qtset/{}_rgb.ftmp'.format(idx+1))
    depth.tofile('./icraft/qtset/{}_depth.ftmp'.format(idx+1))
    if idx == 15:
        break