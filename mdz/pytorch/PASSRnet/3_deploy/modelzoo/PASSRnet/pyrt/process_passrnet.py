import torch
import cv2
import numpy as np
import math
import skimage
#===== 网络超参数 ======#
INTERVAL = 200

def read_list_from_txt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

def cal_psnr(img1, img2):
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    # return measure.compare_psnr(img1_np, img2_np)#老版本写法，给一个图片计算峰值信噪比
    return skimage.metrics.peak_signal_noise_ratio(img1_np,img2_np)

