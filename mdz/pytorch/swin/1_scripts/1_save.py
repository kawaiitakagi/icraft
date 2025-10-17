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

from iutils.config4icraft import get_config
from models import build_model
from logger import create_logger

from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor
from models.swin_transformer import WindowAttention,SwinTransformer
WEIGHTS_PATH = "../weights/swin_tiny_patch4_window7_224.pth"

IMG_PATH = '../2_compile/qtset/imagenet/ILSVRC2012_val_00000002.JPEG'
IMG_H = 224
IMG_W = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_option():

    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    # parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument('--cfg', type=str, default="configs/swin/swin_tiny_patch4_window7_224.yaml", metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path',default="", type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',default=WEIGHTS_PATH,
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume',help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    # parser.add_argument('--eval', default="" action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval', default=True ,type=bool, help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", default= 0 ,type=int, required=False, help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

transform = transforms.Compose([
        transforms.Resize(size=256, max_size=None, antialias=None),
        transforms.CenterCrop(size=(IMG_H, IMG_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
    )

def main(config):
    logger = create_logger(output_dir="")

    model = build_model(config).to(device)
    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model, logger)
    model.eval()
    image = Image.open(IMG_PATH, mode='r')
    img = transform(image).unsqueeze(dim=0).to(device)
    out = model(img)
    print(out.shape)
    print(out.argmax(1))
    print(index2label[int(out.argmax(1)[0])])
    torch.onnx.export(model.cpu(), img.cpu(),"../2_compile/fmodel/swin_tiny_"+str(IMG_H)+"x"+str(IMG_W)+".onnx",opset_version=17)
    print("Trace Done ! Traced model is saved to "+"../2_compile/fmodel/swin_tiny_"+str(IMG_H)+"x"+str(IMG_W)+".onnx")
    # trc_model = torch.jit.trace(model.cpu(), img.cpu())
    # trc_model.save(TRACED_MODEL_PATH)
    # print('traced done')

if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    # torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    # torch.distributed.barrier()

    # seed = config.SEED + dist.get_rank()
    seed = config.SEED 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE  / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    main(config)
