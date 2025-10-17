WEIGHTS_PATH = "../weights/deeplab-resnet.pth.tar"
IMAGE_PATH = "../2_compile/qtset/voc2012/2012_004266.jpg"
import os
import sys
sys.path.append(R"../0_deeplabv3plus")
import os
from modeling.deeplab import *
import argparse
from dataloaders import make_data_loader
from tqdm import tqdm
from dataloaders import custom_transforms as tr
from torchvision import transforms
from PIL import Image
import torch
import numpy as np


def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap
cmap = voc_cmap()

def decode_target(mask):
    """decode semantic mask to RGB image"""
    return cmap[mask]
if __name__ == "__main__":

    model = DeepLab(num_classes=21,
                    backbone='resnet',
                    output_stride=16,
                    sync_bn=None,
                    freeze_bn=False)

    checkpoint = torch.load(WEIGHTS_PATH,map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    composed_transforms = transforms.Compose([
        tr.FixScaleCrop(crop_size=513),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])
    _img = Image.open(IMAGE_PATH).convert('RGB')
    sample = {'image': _img, 'label': _img}
    convert_img = composed_transforms(sample)
    output = model(convert_img['image'].unsqueeze(0))
    print(output.shape)
    preds = output.max(1)[1].detach().cpu().numpy()
    colorized_preds = decode_target(preds).astype('uint8')
    colorized_preds = Image.fromarray(colorized_preds[0]) # to PIL Image
    colorized_preds.show()