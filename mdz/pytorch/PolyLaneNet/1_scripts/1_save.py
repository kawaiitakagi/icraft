# python .\1_save.py --cfg .\config.yaml --exp_name tusimple
import os
import sys
import random
import logging
import argparse
import subprocess
from time import time

import numpy as np
import torch
sys.path.append(R"../0_PolyLaneNet")
from lib.config import Config
from lib.models import PolyRegression

def new_forward(self, x, epoch=None, **kwargs):
    output, extra_outputs = self.model(x, **kwargs)
    for i in range(len(self.curriculum_steps)):
        if epoch is not None and epoch < self.curriculum_steps[i]:
            output[:, -len(self.curriculum_steps) + i] = 0
    # return output,extra_outputs 
    return output # for freeze pb

PolyRegression.forward = new_forward 


def test(model, exp_root, input_size,pt_path,epoch, verbose=True):
    if verbose:
        logging.info("Starting tracing model.")

    # Test the model
    if epoch > 0:
        model.load_state_dict(torch.load(os.path.join(exp_root, "models", "model_{:03d}.pt".format(epoch)),map_location=torch.device('cpu'))['model'])

    model.eval()
    # ===============================Freeze Pb================================
    images = torch.ones((1,3,input_size[0],input_size[1]))
    script_model = torch.jit.trace(model,images)
    torch.jit.save(script_model,pt_path)
    logging.info(f"Export Done!Traced model save at:{pt_path}")


def log_on_exception(exc_type, exc_value, exc_traceback):
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lane regression")
    parser.add_argument("--exp_name", default="tusimple", help="Experiment name")
    parser.add_argument("--cfg", default="config.yaml", help="Config file")
    parser.add_argument("--epoch", type=int, default=2695, help="Epoch to test the model on")
    parser.add_argument("--batch_size", type=int, help="Number of images per batch")
    parser.add_argument("--save_path", type=str,default="../2_compile/fmodel/PolyLaneNet_360x640.pt", help="model path(s) for inference.")
    parser.add_argument('--imgsz', nargs='+', type=tuple, default=(360,640), help='(input height,input width)')
    args = parser.parse_args()
    cfg = Config(args.cfg)
    input_size = args.imgsz
    save_path = args.save_path
    
    
    # Set up seeds
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])

    # Set up logging
    exp_root = os.path.join(cfg['exps_dir'], os.path.basename(os.path.normpath(args.exp_name)))

    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(exp_root, "test_log.txt")),
            logging.StreamHandler(),
        ],
    )

    sys.excepthook = log_on_exception

    logging.info("Experiment name: {}".format(args.exp_name))
    logging.info("Config:\n" + str(cfg))
    logging.info("Args:\n" + str(args))

    # Device configuration
    device = torch.device("cpu")

    # Hyper parameters
    num_epochs = cfg["epochs"]
    batch_size = cfg["batch_size"] if args.batch_size is None else args.batch_size

    # Model
    model = cfg.get_model().to(device)
    
    test_epoch = args.epoch

    test(model, exp_root, input_size, save_path,epoch=test_epoch)
