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
from kornia import tensor_to_image, image_to_tensor
from kornia.color import ycbcr_to_rgb, rgb_to_bgr, rgb_to_ycbcr, bgr_to_rgb
import torch
from torchvision.transforms import Resize


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


def gray_read(img_path):
    img_n = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    img_t = image_to_tensor(img_n).float() / 255
    return img_t


def ycbcr_read(img_path):
    img_n = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    img_t = image_to_tensor(img_n).float() / 255
    img_t = rgb_to_ycbcr(bgr_to_rgb(img_t))
    y, cbcr = torch.split(img_t, [1, 2], dim=0)
    return y, cbcr

ip_octet = 93
GENERATED_JSON_FILE = "../2_compile/imodel/tardal_dt/tardal_dt_BY.json"
GENERATED_RAW_FILE = "../2_compile//imodel/tardal_dt/tardal_dt_BY.raw"

generated_network = Network.CreateFromJsonFile(GENERATED_JSON_FILE)
generated_network.loadParamsFromFile(GENERATED_RAW_FILE)


vi_path = '../0_TarDAL-main/data/m3fd/vi/00000.png'
ir_path = '../0_TarDAL-main/data/m3fd/ir/00000.png'
ori_img = cv2.imread(vi_path)
vi, cbcr = ycbcr_read(vi_path)
ir = gray_read(ir_path)
max_size = torch.Size([768, 1024])
transform_fn = Resize(size=max_size)
t = torch.cat([ir, vi, cbcr], dim=0)
ir, vi, cbcr = torch.split(transform_fn(t), [1, 1, 2], dim=0)
ir = ir.unsqueeze(-1).numpy().transpose(3, 1, 2, 0)
vi = vi.unsqueeze(-1).numpy().transpose(3, 1, 2, 0)
cbcr = cbcr.unsqueeze(-1).numpy().transpose(3, 1, 2, 0)
# ir.tofile('./ir.bin')
# vi.tofile('./vi.bin')
# cbcr.tofile('./cbcr.bin')

# ir = np.fromfile('./ir.bin', dtype=np.float32).reshape((1, 768, 1024, 1))
# vi = np.fromfile('./vi.bin', dtype=np.float32).reshape((1, 768, 1024, 1))
# cbcr = np.fromfile('./cbcr.bin', dtype=np.float32).reshape((1, 768, 1024, 2))
input1, input2 = Tensor(ir, Layout("NHWC")), Tensor(vi, Layout("NHWC"))
generated_output = run(generated_network, [input1, input2])
output = generated_output[0]

fuse = np.concatenate((output, cbcr), axis=3).transpose(0, 3, 1, 2)
# fuse.tofile('./fuse.bin')
# fuse = np.fromfile('./fuse.bin', dtype=np.float32).reshape((1, 768, 1024, 3)).transpose(0, 3, 1, 2)
img = ycbcr_to_rgb(torch.from_numpy(fuse))
img = rgb_to_bgr(img)
img_n = tensor_to_image(img.squeeze().cpu()) * 255
if ori_img.shape != img_n.shape:
    img_n = cv2.resize(img_n, (ori_img.shape[1], ori_img.shape[0]), interpolation=cv2.INTER_LINEAR)
# cv2.imwrite('./fuse.png', img_n)
cv2.imshow(" ", img_n)
cv2.waitKey(0)