
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import sys
sys.path.append("../0_swin")
import os
import time
import json
import random
import argparse
import datetime
import numpy as np
from iutils.index2label import index2label
# from torchsummary import summary
from PIL import Image
import torch
import torchvision
from torchvision import datasets, transforms
import onnxruntime

from models import build_model
from logger import create_logger

from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor
from models.swin_transformer import WindowAttention,SwinTransformer
TRACED_MODEL_PATH = '../2_compile/fmodel/swin_tiny_224x224.onnx'
IMG_PATH = '../2_compile/qtset/imagenet/ILSVRC2012_val_00000002.JPEG'
IMG_H = 224
IMG_W = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
        transforms.Resize(size=256, max_size=None, antialias=None),
        transforms.CenterCrop(size=(IMG_H, IMG_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
    )
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
def main():
    logger = create_logger(output_dir="")
    image = Image.open(IMG_PATH, mode='r')
    img = transform(image).unsqueeze(dim=0).to(device)
    model = onnxruntime.InferenceSession(TRACED_MODEL_PATH,providers=['CPUExecutionProvider'])
    # model = torch.jit.load(TRACED_MODEL_PATH).to(device)
    # model.eval()
    inputs = {model.get_inputs()[0].name: to_numpy(img)}
    out = model.run(None, inputs)
    # out = model(img)
    # print(out.shape)
    print(torch.Tensor(np.array(out)).argmax(2))
    print(index2label[int(torch.Tensor(np.array(out)).argmax(2)[0])])


if __name__ == '__main__':
    main()
