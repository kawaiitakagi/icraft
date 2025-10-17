import sys
sys.path.append("./tools")
import argparse
import os
import torch
import _init_paths
import models
import datasets
from config import config
from config import update_config
from utils.utils import create_logger
from torch.nn import functional as F
# import  ddrnet_23_test 
# from ddrnet_23_test import DualResNet
from lib.models.ddrnet_23 import *

RED = '\033[31m'       # 设置前景色为红色
RESET = '\033[0m'      # 重置所有属性到默认值
ver = torch.__version__
assert ("1.6" in ver) or ("1.9" in ver), f"{RED}Unsupported PyTorch version: {ver}{RESET}"

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="experiments/cityscapes/ddrnet23.yaml",
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--weight", default="../weights/best_val.pth", type=str, help="trained model path")
    parser.add_argument("--export_dir", type=str, default="../2_compile/fmodel/", help="path of traced model")
    args = parser.parse_args()
    update_config(config, args)
    return args
   

def new_forward(self, x):

    layers = []
    x = self.conv1(x)
    x = self.layer1(x)
    layers.append(x)

    x = self.layer2(self.relu(x))
    layers.append(x)

    x = self.layer3(self.relu(x))
    layers.append(x)
    x_ = self.layer3_(self.relu(layers[1]))

    x = x + self.down3(self.relu(x_))
    x_ = x_ + F.interpolate(
                    self.compression3(self.relu(layers[2])),
                    size=[OUT_H, OUT_W],
                    mode='bilinear')
    if self.augment:
        temp = x_

    x = self.layer4(self.relu(x))
    layers.append(x)
    x_ = self.layer4_(self.relu(x_))

    x = x + self.down4(self.relu(x_))
    x_ = x_ + F.interpolate(
                    self.compression4(self.relu(layers[3])),
                    size=[OUT_H, OUT_W],
                    mode='bilinear')

    x_ = self.layer5_(self.relu(x_))
    x = F.interpolate(
                    self.spp(self.layer5(self.relu(x))),
                    size=[OUT_H, OUT_W],
                    mode='bilinear')

    x_ = self.final_layer(x + x_)

    if self.augment: 
        x_extra = self.seghead_extra(temp)
        return [x_extra, x_]
    else:
        return x_  
   

# 全局变量
OUT_W = 1
OUT_H= 1

if __name__ == '__main__':

    args = parse_args()
    logger, final_output_dir, _ = create_logger(config, args.cfg, 'test')

    # build model
    # if torch.__version__.startswith('1'):
    #     module = eval('ddrnet_23_test')
    #     module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
        
    # model = eval('ddrnet_23_test' + '.get_seg_model')(config)
    model = DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=19, planes=64, spp_planes=128, head_planes=128, augment=False)

    # load model
    model_state_file = args.weight  
    logger.info('=> loading model from {}'.format(model_state_file))
    pretrained_dict = torch.load(model_state_file, map_location=torch.device("cpu"))
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()

    # export model
    if not os.path.exists(args.export_dir):
        os.makedirs(args.export_dir)
    input = torch.rand((1, 3, 1024, 2048))
    OUT_W = input.shape[-1] // 8
    OUT_H = input.shape[-2] // 8
    DualResNet.forward = new_forward
    trcnet = torch.jit.trace(model, input)
    # _ = trcnet(input)
    trcnet.save(args.export_dir+"DDRNet_1024x2048_traced.pt")
    print("successful save model in ", args.export_dir+"DDRNet_1024x2048_traced.pt")
