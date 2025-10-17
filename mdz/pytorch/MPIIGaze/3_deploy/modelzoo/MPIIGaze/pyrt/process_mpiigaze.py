
import argparse
import torch
import yacs.config
import sys
sys.path.append(R"../../../../0_MPIIGaze")
from gaze_estimation.config import get_default_config

def load_config() -> yacs.config.CfgNode:
    configfn = "config/demo_mpiigaze_resnet.yaml"
    config = get_default_config()
    config.merge_from_file(configfn)
    if not torch.cuda.is_available():
        config.device = 'cpu'
        config.train.train_dataloader.pin_memory = False
        config.train.val_dataloader.pin_memory = False
        config.test.dataloader.pin_memory = False
    config.freeze()
    return config


