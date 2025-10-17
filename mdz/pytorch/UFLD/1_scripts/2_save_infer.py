# test a single image 
import numpy as np
import torch
import cv2
import os
import os.path as osp
import glob
import argparse
import sys 

from utils import get_lanes,to_array,imshow_lanes
sys.path.append(R"../0_UFLD")
from lanedet.datasets.process import Process
from lanedet.utils.config import Config
from pathlib import Path
from tqdm import tqdm

class Detect(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.processes = Process(cfg.val_process, cfg)
        self.pt_path = cfg.load_from

    def preprocess(self, img_path):
        ori_img = cv2.imread(img_path)
        img = ori_img[self.cfg.cut_height:, :, :].astype(np.float32)
        data = {'img': img, 'lanes': []}
        data = self.processes(data)
        data['img'] = data['img'].unsqueeze(0)
        data.update({'img_path':img_path, 'ori_img':ori_img})
        return data

    def inference(self, data):
        data = data['img']
        
        model = torch.jit.load(self.pt_path)
        out = model(data)
        out = out.reshape((-1,101,56,6)) # cpu reshape to target size (1,101,56,6)
        return out

    def post_process(self,out):
        
        out = out.detach().numpy()
        lanes = get_lanes(out)
        ori_lane_lists = []
        for lane in lanes:
            ori_lane = to_array(lane)
            ori_lane_lists.append(ori_lane)
        return  ori_lane_lists   


    def run(self, data):
        data = self.preprocess(data)
        out = self.inference(data)
        
        ori_lane_lists = self.post_process(out)
        
        out_file = self.cfg.savedir
        out_file = osp.join(out_file, osp.basename(data['img_path']))
        if self.cfg.savedir:
            imshow_lanes(data['ori_img'],ori_lane_lists,out_file=out_file)
        if self.cfg.show:
            result_image = cv2.imread(out_file)
            cv2.imshow('result_image',result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return data

def get_img_paths(path):
    p = str(Path(path).absolute())  # os-agnostic absolute path
    print('****p =',p)
    if '*' in p:
        paths = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        paths = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
        paths = [p]  # files
    else:
        raise Exception(f'ERROR: {p} does not exist')
    return paths 

def process(args):
    cfg = Config.fromfile(args.config)
    cfg.show = args.show
    cfg.savedir = args.savedir
    cfg.load_from = args.load_from
    detect = Detect(cfg)
    paths = get_img_paths(args.img)
    for p in tqdm(paths):
        detect.run(p)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default="../0_UFLD/configs/ufld/resnet18_tusimple.py", help='The path of config file')
    parser.add_argument('--img',default="./images/tusimple/",  help='The path of the img (img file or img_folder), for example: data/*.png')
    parser.add_argument('--show', action='store_true', help='Whether to show the image')
    parser.add_argument('--savedir', type=str, default="./pt_vis/tusimple", help='The root of save directory')
    parser.add_argument('--load_from', type=str, default='../2_compile/fmodel/UFLD_288x800.pt', help='The path of model')
    args = parser.parse_args()
    print('args = ',args)
    process(args)
    print('lane results save at',args.savedir)
