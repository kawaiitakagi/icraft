import sys
sys.path.append(R"../0_JL-DCF-pytorch-master")
import argparse
import os
import time
import torch
from torch.nn import functional as F
import cv2
import numpy as np


def Normalization(image):
    in_ = image[:, :, ::-1]
    in_ -= np.array((123.675, 116.28, 103.53))
    in_ /= np.array((58.395, 57.12, 57.375))
    return in_


def load_image(im, image_size):
    in_ = np.array(im, dtype=np.float32)
    in_ = cv2.resize(in_, (image_size, image_size))
    in_ = Normalization(in_)
    in_ = in_.transpose((2, 0, 1))
    return in_


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = torch.jit.load('./JLDCF.pt')
net.to(device)
net.eval()

img_path = '../0_JL-DCF-pytorch-master/dataset/test/LFSD/RGB/1.jpg'
dep_path = '../0_JL-DCF-pytorch-master/dataset/test/LFSD/depth/1.png'

image_size = 320

images = cv2.imread(img_path)
depth = cv2.imread(dep_path)
im_size = tuple(images.shape[:2])
images = np.ascontiguousarray(load_image(images, image_size))
depth = np.ascontiguousarray(load_image(depth, image_size))
images = torch.from_numpy(images).unsqueeze(0)
depth = torch.from_numpy(depth).unsqueeze(0)

with torch.no_grad():
    images = images.to(device)
    depth = depth.to(device)

    preds = net(images, depth)

    preds = F.interpolate(preds, (im_size[0], im_size[1]), mode='bilinear', align_corners=True)
    pred = np.squeeze(torch.sigmoid(preds)).cpu().data.numpy()

    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    multi_fuse = 255 * pred

cv2.imwrite('test.png', multi_fuse)