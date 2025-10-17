import torch
import cv2
import numpy as np
import math
import re
#===== 网络超参数 ======#
INTERVAL = 200

def read_list_from_txt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

def numeric_sort_key(s):
    return int(re.search(r'\d+', s).group())

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x[0]
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return np.array([b], dtype=np.float32)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def save_bb(file, data):
    tracked_bb = np.array(data).astype(int)
    np.savetxt(file, tracked_bb, delimiter='\t', fmt='%d')

def clip_box(box: list, H, W, margin=0):
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0, x1), W-margin)
    x2 = min(max(margin, x2), W)
    y1 = min(max(0, y1), H-margin)
    y2 = min(max(margin, y2), H)
    w = max(margin, x2-x1)
    h = max(margin, y2-y1)
    return [x1, y1, w, h]

def map_box_back(state, pred_box: list, resize_factor: float, search_size):
    cx_prev, cy_prev = state[0] + 0.5 * state[2], state[1] + 0.5 * state[3]
    cx, cy, w, h = pred_box
    half_side = 0.5 * search_size / resize_factor
    cx_real = cx + (cx_prev - half_side)
    cy_real = cy + (cy_prev - half_side)
    return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

def sample_target(im, bb, z_factor, output_sz=None):
    x, y, w, h = bb
    # Crop image
    crop_sz = math.ceil(math.sqrt(w * h) * z_factor)
    if crop_sz < 1:
        raise Exception('Too small bounding box.')
    x1 = int(round(x + 0.5 * w - crop_sz * 0.5))
    x2 = int(x1 + crop_sz)

    y1 = int(round(y + 0.5 * h - crop_sz * 0.5))
    y2 = int(y1 + crop_sz)

    x1_pad = int(max(0, -x1))
    x2_pad = int(max(x2 - im.shape[1] + 1, 0))

    y1_pad = int(max(0, -y1))
    y2_pad = int(max(y2 - im.shape[0] + 1, 0))
    # Crop target
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
    # Pad
    im_crop_padded = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT)
    if output_sz is not None:
        resize_factor = output_sz / crop_sz
        im_crop_padded = cv2.resize(im_crop_padded, (output_sz, output_sz))
        return im_crop_padded, resize_factor
    else:
        return im_crop_padded, 1.0

def preprocess(img_arr:np.ndarray, mean, std):
    img_tensor = torch.tensor(img_arr).float().permute((2,0,1)).unsqueeze(dim=0)
    img_tensor_norm = ((img_tensor / 255.0) - mean) / std  # (1,3,H,W)
    return img_tensor_norm

def postprocess(output_tensor,resize_factor,s_img,state,SEARCH_SIZE):
    coord_l, coord_t, coord_r, coord_b, pred_scores_feat = output_tensor
    pred_boxes = np.stack((np.array(coord_l), np.array(coord_t), np.array(coord_r), np.array(coord_b)), axis=1).reshape(-1, 4) / SEARCH_SIZE
    pred_boxes = box_xyxy_to_cxcywh(pred_boxes)
    pred_scores_feat = np.array(pred_scores_feat)
    pred_score = sigmoid(pred_scores_feat.mean(axis=1).reshape(-1)[0])
    pred_box = (np.mean(pred_boxes, axis=0) * SEARCH_SIZE / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
    H, W = s_img.shape[:2]
    state = clip_box(map_box_back(state, pred_box, resize_factor, SEARCH_SIZE), H, W, margin=10)#更新state
    return pred_score,pred_box,state