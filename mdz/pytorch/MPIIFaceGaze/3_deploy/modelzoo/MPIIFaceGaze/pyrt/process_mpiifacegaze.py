
import argparse
import torch
import yacs.config
import sys
sys.path.append(R"../../../../0_MPIIFaceGaze")
from gaze_estimation.config import get_default_config

def load_config() -> yacs.config.CfgNode:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default="config/demo_mpiifacegaze_resnet_simple_14.yaml", type=str)
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    if args.config is not None:
        config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    if not torch.cuda.is_available():
        config.device = 'cpu'
        config.train.train_dataloader.pin_memory = False
        config.train.val_dataloader.pin_memory = False
        config.test.dataloader.pin_memory = False
    config.freeze()
    return config


