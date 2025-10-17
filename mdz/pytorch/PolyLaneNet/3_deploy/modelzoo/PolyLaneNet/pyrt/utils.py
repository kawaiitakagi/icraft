import yaml
import torch
import torch.nn as nn
import cv2
import numpy as np

def preprocess(ori_img,IMG_W = 640,IMG_H = 360):
    # Resize img to (360,640,3)
    img = cv2.resize(ori_img,(IMG_W,IMG_H),interpolation=cv2.INTER_LINEAR)
    img = np.expand_dims(img,axis=0)
    return img 

def post_process(outputs):
    pred = outputs
    s = nn.Sigmoid()
    #target.shape[1] = 5
    
    # pred = pred.reshape(-1, target.shape[1], 1 + 2 + 4)
    pred = pred.reshape(1, -1, 1 + 2 + 4)

    pred_confs = s(pred[:, :, 0]).reshape((-1, 1))
    pred_uppers =  pred[:, :, 2].reshape((-1, 1))
    pred_polys =  pred[:, :, 3:].reshape(-1, 4)
    pred_lowers = pred[:, :, 1]
    pred_lowers[...] = pred_lowers[:, 0].reshape(-1, 1).expand(pred.shape[0], pred.shape[1])
    return pred 

def decode(all_outputs, conf_threshold=0.5):
    outputs= all_outputs
    
    outputs = outputs.reshape(len(outputs), -1, 7)  # score + upper + lower + 4 coeffs = 7

    outputs[:, :, 0] = torch.sigmoid(outputs[:, :, 0])
    outputs[outputs[:, :, 0] < conf_threshold] = 0

    # if False and self.share_top_y:
    #     outputs[:, :, 0] = outputs[:, 0, 0].expand(outputs.shape[0], outputs.shape[1])

    return outputs
