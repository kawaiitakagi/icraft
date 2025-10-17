# import torch

from icraft.xir import *
from icraft.xrt import *
from icraft.host_backend import *
from icraft.buyibackend import *

import sys
import cv2
import numpy as np
import os
import json
from yolov9_utils import (DFL, make_anchors, dist2bbox, sigmoid, non_max_suppression, 
                          COCO_CLASSES, scale_boxes, process_mask, box_label, colors,
                          draw_masks, panoptic_merge_show, scale_image)

# Compile the model without customop(ie: detpost) 
GENERATED_JSON_FILE = "../3_deploy/modelzoo/yolov9_pan/imodel/16/gelan_c_pan_BY.json"
GENERATED_RAW_FILE = "../3_deploy/modelzoo/yolov9_pan/imodel/16/gelan_c_pan_BY.raw"
IMG_PATH = "../0_yolov9/data/images/horses.jpg"

# load model
generated_network = Network.CreateFromJsonFile(GENERATED_JSON_FILE)
generated_network.loadParamsFromFile(GENERATED_RAW_FILE)
print('INFO: Create network!')

# # CV2 preprocessing # from 'letterbox'
img_raw = cv2.imread(IMG_PATH)
def preprocess_image(img_raw, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img_raw.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(img_raw, new_unpad, interpolation=cv2.INTER_LINEAR)
    else:
        im = img_raw
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    
    img = im[:,:,::-1].copy() # BGR to RGB
    
    img_ = np.expand_dims(img,axis=0)
    img_ = np.ascontiguousarray(img_)
    input_tensor = Tensor(img_, Layout("NHWC"))
    return input_tensor, ratio, (dw, dh)

image, ratio, (dw, dh) = preprocess_image(img_raw)

# create forward Session
session = Session.Create([HostBackend],generated_network.view(0),[HostDevice.Default()])
session.apply()
generated_output = session.forward([image])
# 11 out  in n,h,w,c format
# i = 0  out = (1, 80, 80, 80) 
# i = 1  out = (1, 80, 80, 64) 
# i = 2  out = (1, 80, 80, 32) 
# i = 3  out = (1, 40, 40, 80) 
# i = 4  out = (1, 40, 40, 64) 
# i = 5  out = (1, 40, 40, 32) 
# i = 6  out = (1, 20, 20, 80) 
# i = 7  out = (1, 20, 20, 64) 
# i = 8  out = (1, 20, 20, 32) 
# i = 9  out = (1, 160, 160, 32) 
# i = 10  out = (1, 160, 160, 173) 
# check outputs
for i in range(len(generated_output)):
    out = np.array(generated_output[i])
    print(out.shape)
print('INFO: get forward results!')

proto = np.array(generated_output[9]).transpose(0, 3, 1, 2) # (1 , 32, 160, 160)

psemasks = np.array(generated_output[10]) # (1, 160, 160, 80 + 93)

# reunion-detect
detect_outputs = []
head = 3
for i in range(head):
    temp1 = np.array(generated_output[3*i]) # cls
    temp2 = np.array(generated_output[3*i+1]) # box
    out = np.concatenate((temp2, temp1), axis=3) # (1, fm, fm, 64+80) 
    out_transposed = out.transpose((0, 3, 1, 2))
    detect_outputs.append(out_transposed)

# postprocess
reg_max = 16
nc = 80
dfl_layer = DFL(reg_max)
anchors, strides = (x.T for x in make_anchors(detect_outputs, np.array([8, 16, 32], dtype=np.float32), 0.5))
# 'split' in numpy is a little different from the same name in pytorch
split_points = [reg_max * 4, reg_max * 4 + nc]
slices = np.split(np.concatenate([xi.reshape(detect_outputs[1].shape[0], reg_max * 4 + 80, -1) for xi in detect_outputs], axis=2), split_points, axis=1)
box, cls = slices[0], slices[1] # [box (1, 64, 8400)], [cls (1, 80, 8400)]
dbox = dist2bbox(dfl_layer.forward(box), np.expand_dims(anchors, axis=0), xywh=True, dim=1) * strides

# Concatenate dbox and cls sigmoid
y = np.concatenate((dbox, sigmoid(cls)), axis=1) # [1, 4+80, 8400]

# Reunion-seg
seg = np.concatenate([np.array(generated_output[2]).transpose(0, 3, 1, 2).reshape(1, 32, -1), 
                      np.array(generated_output[5]).transpose(0, 3, 1, 2).reshape(1, 32, -1), 
                      np.array(generated_output[8]).transpose(0, 3, 1, 2).reshape(1, 32, -1)], axis=2)

# Final prediction
out = np.concatenate((y, seg), axis=1)  # [1, 84+32, 8400]

# NMS
conf_thres = 0.25
iou_thres = 0.45
pred = non_max_suppression(out, conf_thres, iou_thres, classes=None, max_det=1000, nm=32)

masks = process_mask(proto[0], pred[0][:, 6:], pred[0][:, :4], (640,640), upsample=True) # HWn
# remeber the right order: process masks before rescaling boxes !!
pred[0][:, :4] = scale_boxes((640,640), pred[0][:, :4], img_raw.shape).round()  # rescale boxes to im0 size


# Draw pictures
im_pan = img_raw.copy()
for _, (*xyxy, conf, cls) in enumerate(reversed(pred[0][:, :6])):
    c = int(cls) # integer class
    label = f'{COCO_CLASSES[c]} {conf:.2f}'
    im_box = box_label(img_raw, xyxy, label, color=colors(c, True))
im_instance = draw_masks(im_box, masks, colors=[colors(x, True) for x in pred[0][:, 5]])

psemask = cv2.resize(psemasks.squeeze(0), (640, 640), interpolation=cv2.INTER_LINEAR) #shape: [H , W, CLASS], CLASS= 80 + 93 = 173
ns = 173
semantic_mask = psemask.reshape(-1, ns) # (h x w) x class
max_idx = np.argmax(semantic_mask, axis=1)

# merge masks
semask = max_idx.reshape(640, 640)
panoptic = panoptic_merge_show(semask, masks.transpose(2, 0, 1), pred[0][:, 5], pred[0][:, 4],min_area=0)
print("conf: ", pred[0][:, 4])
color_image = np.zeros(panoptic.shape, dtype=np.uint8)
for y in range(panoptic.shape[0]):
    for x in range(panoptic.shape[1]):
        semantic_id = panoptic[y, x, 2]
        if semantic_id != 0:
            color = colors((semantic_id // 1000) + (semantic_id % 1000))
        else:
            color = (0, 0, 0)  # black
        color_image[y, x] = color
color_image = scale_image(color_image.shape[:2], color_image, img_raw.shape)

unique_labels, inverse_indices = np.unique(max_idx, return_inverse=True)
inverse_indices = inverse_indices.reshape(640, 640)
N = unique_labels.size
semantic_masks = np.zeros((N, 640, 640), dtype=np.uint8)
for i in range(N):
    semantic_masks[i] = (inverse_indices == i).astype(np.uint8)

im_semantic = draw_masks(im_pan, semantic_masks.transpose(1, 2, 0), colors=[colors(x, True) for x in unique_labels], alpha=1)

cv2.imshow("semantic window", im_semantic)
cv2.imshow("instance window", im_instance)
cv2.imshow('Panoptic Image', color_image)
cv2.waitKey()







