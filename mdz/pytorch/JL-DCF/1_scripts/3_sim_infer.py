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
import torch
from torch.nn import functional as F


def Normalization(image):
    in_ = image[:, :, ::-1]
    in_ -= np.array((123.675, 116.28, 103.53))
    in_ /= np.array((58.395, 57.12, 57.375))
    return in_


def load_image(im, image_size=320):
    in_ = np.array(im, dtype=np.float32)
    in_ = cv2.resize(in_, (image_size, image_size))
    in_ = Normalization(in_)
    in_ = in_.transpose((2, 0, 1))
    return in_


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
GENERATED_JSON_FILE = "../2_compile/imodel/JLDCF/JLDCF_BY.json"
GENERATED_RAW_FILE = "../2_compile/imodel/JLDCF/JLDCF_BY.raw"

generated_network = Network.CreateFromJsonFile(GENERATED_JSON_FILE)
generated_network.loadParamsFromFile(GENERATED_RAW_FILE)

img_path = '../0_JL-DCF-pytorch-master/dataset/test/LFSD/RGB/1.jpg'
dep_path = '../0_JL-DCF-pytorch-master/dataset/test/LFSD/depth/1.png'
images = cv2.imread(img_path)
shape = images.shape[:2]
depth = cv2.imread(dep_path)
images = np.ascontiguousarray(np.expand_dims(load_image(images).transpose(1, 2, 0), axis=0))
depth = np.ascontiguousarray(np.expand_dims(load_image(depth).transpose(1, 2, 0), axis=0))


generated_output = run(generated_network, [Tensor(images, Layout("NHWC")), Tensor(depth, Layout("NHWC"))])
output = np.array(generated_output[0]).transpose(0, 3, 1, 2)

# output.tofile('output.bin')
# output = np.fromfile('output.bin', dtype=np.float32).reshape(1, 320, 320, 1).transpose(0, 3, 1, 2)
preds = torch.from_numpy(output)

preds = F.interpolate(preds, shape, mode='bilinear', align_corners=True)
pred = np.squeeze(torch.sigmoid(preds)).cpu().data.numpy()

pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
multi_fuse = 255 * pred

cv2.imwrite('./test1.png', multi_fuse)