# python .\0_infer.py --cfg .\config.yaml --exp_name tusimple --view
import os
import sys
import random
import logging
import argparse
import subprocess
from time import time

import cv2
import numpy as np
import torch
import sys 
sys.path.append(R"../0_PolyLaneNet")
from lib.config import Config
from utils.evaluator import Evaluator

def test(model, pt_model,test_loader,  cfg, view,  max_batches=None, verbose=True):
    if verbose:
        logging.info("Starting testing.")

    # Test the model
    criterion_parameters = cfg.get_loss_parameters()
    test_parameters = cfg.get_test_parameters()
    criterion = model.loss
    loss = 0
    total_iters = 0
    test_t0 = time()
    loss_dict = {}
    # predictions = list()
    with torch.no_grad():
        for idx, (images, labels, img_idxs) in enumerate(test_loader):
            if max_batches is not None and idx >= max_batches:
                break
            if idx % 1 == 0 and verbose:
                logging.info("Testing iteration: {}/{}".format(idx + 1, len(test_loader)))
            images = images.to(device)
            labels = labels.to(device)
            
            t0 = time()
            pt_outputs = pt_model(images)
            t = time() - t0
            outputs = (pt_outputs,None)

            loss_i, loss_dict_i = criterion(outputs, labels, **criterion_parameters)
            loss += loss_i.item()
            total_iters += 1
            for key in loss_dict_i:
                if key not in loss_dict:
                    loss_dict[key] = 0
                loss_dict[key] += loss_dict_i[key]

            extra_output = outputs[1]
            outputs = model.decode(outputs, labels, **test_parameters)
            

            if view:
                outputs, extra_outputs = outputs
                # print('outputs =',outputs,type(outputs),outputs[0].shape)
                preds = test_loader.dataset.draw_annotation(
                    idx,
                    pred=outputs[0].cpu().numpy(),
                    cls_pred=extra_outputs[0].cpu().numpy() if extra_outputs is not None else None)
                cv2.imshow('pred', preds)
                cv2.waitKey(0)

    if verbose:
        logging.info("Testing time: {:.4f}".format(time() - test_t0))
    out_line = []
    for key in loss_dict:
        loss_dict[key] /= total_iters
        out_line.append('{}: {:.4f}'.format(key, loss_dict[key]))
    if verbose:
        logging.info(', '.join(out_line))
    
    return  loss / total_iters

def get_code_state():
    state = "Git hash: {}".format(
        subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
    state += '\n*************\nGit diff:\n*************\n'
    state += subprocess.run(['git', 'diff'], stdout=subprocess.PIPE).stdout.decode('utf-8')

    return state


def log_on_exception(exc_type, exc_value, exc_traceback):
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Lane regression")
    parser.add_argument("--cfg", default="config.yaml", help="Config file")
    parser.add_argument("--model_path", default="../2_compile/fmodel/PolyLaneNet_360x640.pt", help="model path(s) for inference.")
    parser.add_argument("--view", type=bool,default = True, help="Show predictions")
    args = parser.parse_args()
    
    cfg = Config(args.cfg)
    
    # Set up seeds
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])

    # Set up logging
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler( "./test_log.txt"),
            logging.StreamHandler(),
        ],
    )

    sys.excepthook = log_on_exception

    logging.info("Config:\n" + str(cfg))
    logging.info("Args:\n" + str(args))

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    # Model
    model = cfg.get_model().to(device)
    pt_path = args.model_path
    pt_model = torch.jit.load(pt_path)
    # Get data set
    test_dataset = cfg.get_dataset("test")

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=1)
    # logging.info('Code state:\n {}'.format(get_code_state()))
    mean_loss = test(model,pt_model, test_loader,  cfg,  view=args.view)
    logging.info("Mean test loss: {:.4f}".format(mean_loss))

   

