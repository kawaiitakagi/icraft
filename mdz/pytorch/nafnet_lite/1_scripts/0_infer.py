TRACED_MODEL_PATH = R"../2_compile/fmodel/Selfmodel-A-selfattn-1c.pt"
INPUT_IMG         = R"../0_nafnet_lite/GOPR0384_11_00-000001.png"
WEIGHTS_PATH      = R'../weights/selfmodel_A_attn.pth'
CFG_FILE          = R'../0_nafnet_lite/options/test/GoPro/Selfmodel_A_selfattn.yml'

import sys
import yaml
import torch
import random
import argparse
import numpy as np
import cv2
import torch.distributed as dist
from os import path as osp
from collections import OrderedDict
sys.path.append(R"../0_nafnet_lite")

from basicsr.models import create_model
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, tensor2img, imwrite
from basicsr.models.archs.Baseline_arch import Baseline
from basicsr.models.archs.NAFNet_arch import NAFNet
from ptflops import get_model_complexity_info

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def parse(opt_path, is_train=True):
    """Parse option file.

    Args:
        opt_path (str): Option file path.
        is_train (str): Indicate whether in training or not. Default: True.

    Returns:
        (dict): Options.
    """
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)

    opt['is_train'] = is_train

    # datasets
    if 'datasets' in opt:
        for phase, dataset in opt['datasets'].items():
            # for several datasets, e.g., test_1, test_2
            phase = phase.split('_')[0]
            dataset['phase'] = phase
            if 'scale' in opt:
                dataset['scale'] = opt['scale']
            if dataset.get('dataroot_gt') is not None:
                dataset['dataroot_gt'] = osp.expanduser(dataset['dataroot_gt'])
            if dataset.get('dataroot_lq') is not None:
                dataset['dataroot_lq'] = osp.expanduser(dataset['dataroot_lq'])

    # paths
    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key
                                  or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(val)
    opt['path']['root'] = osp.abspath(
        osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
    if is_train:
        experiments_root = osp.join(opt['path']['root'], 'experiments',
                                    opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root,
                                                  'training_states')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = osp.join(experiments_root,
                                                'visualization')

        # change some options for debug mode
        if 'debug' in opt['name']:
            if 'val' in opt:
                opt['val']['val_freq'] = 8
            opt['logger']['print_freq'] = 1
            opt['logger']['save_checkpoint_freq'] = 8
    else:  # test
        results_root = osp.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['visualization'] = osp.join(results_root, 'visualization')

    return opt

def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def init_dist(launcher, backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')

def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default= CFG_FILE, required=False, help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--input_path', type=str, default = INPUT_IMG, required=False, help='The path to the input image. For single image inference only.')
    parser.add_argument('--output_path', type=str, default= 'deblur1280_attn_result.png', required=False, help='The path to the output image. For single image inference only.')
    parser.add_argument('--weight_path', type=str, default= WEIGHTS_PATH, required=False, help='The path to weights file.')

    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    if args.input_path is not None and args.output_path is not None:
        opt['img_path'] = {
            'input_img': args.input_path,
            'output_img': args.output_path
        }
    
    opt['path']['pretrain_network_g'] = args.weight_path
    return opt

def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)
    opt['num_gpu'] = torch.cuda.device_count()

    img_path = opt['img_path'].get('input_img')
    output_path = opt['img_path'].get('output_img')

    ## 1. read image
    file_client = FileClient('disk')

    img_bytes = file_client.get(img_path, None)
    try:
        img = imfrombytes(img_bytes, float32=True)
    except:
        raise Exception("path {} not working".format(img_path))
    img = img2tensor(img, bgr2rgb=True, float32=True)

    ## 2. run inference
    opt['dist'] = False
    model = create_model(opt)

    total_parameters = 0
    for name, param in model.net_g.named_parameters():
        total_parameters += param.numel()
    print(f"Total Parameters(M): {total_parameters/1000/1000}")

    img_channel = 3
    width = 16
    dw_expand = 1
    ffn_expand =  1
    enc_blks = [1, 1, 1, 1]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]
    net = Baseline(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                 enc_blk_nums=enc_blks, dec_blk_nums=dec_blks, dw_expand=dw_expand, 
                  ffn_expand=ffn_expand)

    inp_shape = (3, 736, 1280)
    print(inp_shape)

    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    print('flops  :', flops)
    print('params :', params)

    model.feed_data(data={'lq': img.unsqueeze(dim=0)})
    if model.opt['val'].get('grids', False):
        model.grids()
    model.test()

    if model.opt['val'].get('grids', False):
        model.grids_inverse()

    visuals = model.get_current_visuals()
    sr_img = tensor2img([visuals['result']], rgb2bgr=True)

    cv2.imshow(" ", sr_img)
    save_path = "deblur1280_attn_result.png"
    cv2.imwrite(save_path, sr_img)
    cv2.waitKey(0)

    np.save("input_tensor_0infer.npy", img.cpu().numpy())

if __name__ == '__main__':

    main()
