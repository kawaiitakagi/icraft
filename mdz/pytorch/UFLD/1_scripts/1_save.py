# export traced_UFLD model
import numpy as np
import torch
import cv2
import os
import os.path as osp
import glob
import argparse
from lanedet.datasets.process import Process
from lanedet.models.registry import build_net
from lanedet.utils.config import Config
from lanedet.utils.net_utils import load_network
from pathlib import Path
from tqdm import tqdm

# Override LaneCls.forward return tensor, not dict
from lanedet.models.nets.detector import Detector
# Import LaneCls to get_lanes(post-process)
from lanedet.models.heads.lane_cls import LaneCls 
def LaneCls_forward(self, x, **kwargs):
    x = x[-1]
    x = self.pool(x)
    x = x.view(-1, 1800)
    cls = self.cls(x)
    return cls 
LaneCls.forward = LaneCls_forward

def Detector_forward(self, batch):
    output = {}
    fea = self.backbone(batch)  # for freeze pt

    if self.aggregator:
        fea[-1] = self.aggregator(fea[-1])

    if self.neck:
        fea = self.neck(fea)

    if self.training:
        out = self.heads(fea, batch=batch)
        output.update(self.heads.loss(out, batch))
    else:
        output = self.heads(fea)        
    return output
Detector.forward = Detector_forward

class Detect(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.processes = Process(cfg.val_process, cfg)
        self.net = build_net(self.cfg)
        self.net = torch.nn.parallel.DataParallel(
                self.net, device_ids = range(1)).cuda()
        self.net.eval()
        load_network(self.net, self.cfg.load_from)

def process(args):
    cfg = Config.fromfile(args.config)
    cfg.savedir = args.savedir
    cfg.load_from = args.load_from
    detect = Detect(cfg)
    # ===================Freeze Pb=========================
    fake_input = torch.ones((1,3,args.imgsz[0],args.imgsz[1]))
    func = detect.net.module.cpu()
    script_model = torch.jit.trace(func,fake_input,strict=False)
    torch.jit.save(script_model,cfg.savedir)
    print('Export Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default="../0_UFLD/configs/ufld/resnet18_tusimple.py", help='The path of config file')
    parser.add_argument('--savedir', type=str, default="../2_compile/fmodel/UFLD_288x800.pt", help='The root of save directory')
    parser.add_argument('--load_from', type=str, default='../weights/ufld_r18_tusimple.pth', help='The path of model')
    parser.add_argument('--imgsz', type=tuple, default=(288,800), help='image size')
    args = parser.parse_args()
    print('args = ',args)
    process(args)
    print('traced model save at',args.savedir)
