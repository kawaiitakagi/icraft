import sys
sys.path.append(R"../0_TarDAL-main")
import argparse
import logging
from pathlib import Path
import os
import torch
import torch.backends.cudnn
from torch.utils.data import DataLoader
import yaml
import numpy as np
from kornia.color import ycbcr_to_rgb

import scripts
from config import from_dict
from module.fuse.generator import Generator
import loader
from tools.dict_to_device import dict_to_device
from kornia import image_to_tensor, tensor_to_image
from kornia.color import rgb_to_ycbcr, bgr_to_rgb, rgb_to_bgr
import cv2
from tqdm import tqdm

if __name__ == '__main__':
    # args parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='../0_TarDAL-main/config/official/infer/tardal-dt.yaml', help='config file path')
    parser.add_argument('--save_dir', default='../0_TarDAL-main/runs/tmp', help='fusion result save folder')
    args = parser.parse_args()
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # init config
    config = yaml.safe_load(Path(args.cfg).open('r'))
    config = from_dict(config)  # convert dict to object
    config = config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    infer_p = getattr(scripts, 'InferF')

    data_t = getattr(loader, config.dataset.name)
    p_dataset = data_t(root=config.dataset.root, mode='pred', config=config)

    p_loader = DataLoader(
            p_dataset, batch_size=config.inference.batch_size, shuffle=False,
            collate_fn=data_t.collate_fn, pin_memory=True, num_workers=config.inference.num_workers,
        )

    f_dim, f_depth = config.fuse.dim, config.fuse.depth
    generator = Generator(dim=f_dim, depth=f_depth)
    f_ckpt = '../0_TarDAL-main/weights/tardal-dt.pth'
    ckpt = torch.load(f_ckpt, map_location='cpu')
    f_ckpt = ckpt
    f_ckpt.pop('use_eval')
    generator.load_state_dict(f_ckpt)
    # print(generator)
    generator.to(device)
    generator.eval()
    # print(config.inference.batch_size)
    idx = 0
    for sample in tqdm(p_loader):
        sample = dict_to_device(sample, device)
        # print(sample['ir'].size())
        # print(sample['vi'].size())
        ir, vi = sample['ir'], sample['vi']
        ir_input = ir.cpu().numpy().transpose(0, 2, 3, 1)
        vi_input = vi.cpu().numpy().transpose(0, 2, 3, 1)
        # cbcr = sample['cbcr'].cpu().numpy().transpose(0, 2, 3, 1)
        # ir_input.tofile('dumps/ir.bin')
        # vi_input.tofile('dumps/vi.bin')
        ir_input.tofile('qtset/ir_{}.ftmp'.format(sample['name'][0].split('.')[0]))
        vi_input.tofile('qtset/vi_{}.ftmp'.format(sample['name'][0].split('.')[0]))
        # cbcr.tofile('qtset/cbcr_{}.bin'.format(sample['name'][0].split('.')[0]))
        # cbcr = sample['cbcr'].cpu().numpy().transpose(0, 2, 3, 1).tofile('dumps/cbcr.bin')
        # print(ir_input.shape)
        # print(data_t.color, config.inference.grayscale)
        # print(sample['cbcr'].shape)
        # print(sample['shape'])
        # print(sample['name'])
        # with torch.no_grad():
        #     fus = generator(ir, vi)
        # print(fus.cpu().numpy().shape)
        # fus = torch.cat([fus, sample['cbcr']], dim=1)
        # fus = ycbcr_to_rgb(fus)
        # print(fus.size())
        # img_t = rgb_to_bgr(fus)
        # img_n = tensor_to_image(img_t.squeeze().cpu()) * 255
        # cv2.imwrite('./temp.png', img_n)
        idx += 1
        if idx >= 16:
            break