import cv2
import numpy as np
import re
#===== 网络超参数 ======#
WINDOW_INFLUENCE = 0.225  # from pytrcking/tracker/et_tracker.py->class Config
LR = 0.616
PENALTY_K =0.007 # 超参 # from pytrcking/tracker/et_tracker.py->class Config
CONTEXT_AMOUNT = 0.5
RATIO = 1
WINDOWING = 'cosine'
EXEMPLAR_SIZE = 127

def grids(score_size=16,total_stride=16,instance_size=256):
    """
    each element of feature map on input search image
    :return: H*W*2 (position for each element)
    """
    # print('ATTENTION',instance_size,score_size)
    # the real shift is -param['shifts']
    sz_x = score_size // 2
    sz_y = score_size // 2

    x, y = np.meshgrid(np.arange(0, score_size) - np.floor(float(sz_x)),
                        np.arange(0, score_size) - np.floor(float(sz_y)))
    grid_to_search_x = x * total_stride + instance_size // 2
    grid_to_search_y = y * total_stride + instance_size // 2
    return grid_to_search_x, grid_to_search_y

def get_subwindow_tracking(image, target_pos, model_sz, original_sz, avg_chans):
    # crop_img
    c = (original_sz+1) / 2
    context_xmin = round((target_pos[0] - c).item())
    context_xmax = context_xmin + original_sz - 1
    context_ymin = round((target_pos[1] - c).item())
    context_ymax = context_ymin + original_sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - image.shape[1] + 1))
    bottom_pad = int(max(0., context_ymax - image.shape[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    r, c, k = image.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)
        # for return mask
        # tete_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad))

        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = image
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        # tete_im = np.zeros(img_sz.shape[0:2])
        im_patch_original = image[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
        
    im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
    return im_patch

# ---------------------------------
# Functions for FC tracking tools
# ---------------------------------
def python2round(f):
    """
    use python2 round function in python3
    """
    if round(f + 1) - round(f) != 1:
        return f + abs(f) / f * 0.5
    return round(f)

def net1_preprocess(image, target_pos, target_sz):
    wc_z = target_sz[0] + CONTEXT_AMOUNT * sum(target_sz)
    hc_z = target_sz[1] + CONTEXT_AMOUNT * sum(target_sz)
    s_z = round(np.sqrt(wc_z * hc_z).item())
    avg_chans = np.mean(image, axis=(0, 1))#每通道均值
    z_crop = get_subwindow_tracking(image, target_pos, EXEMPLAR_SIZE, s_z, avg_chans)
    return z_crop, avg_chans

def net2_preprocess(image,target_pos,target_sz, instance_size, avg_chans):
    wc_z = target_sz[0] + CONTEXT_AMOUNT * sum(target_sz)
    hc_z = target_sz[1] + CONTEXT_AMOUNT * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)
    #与net1区别：
    scale_z = EXEMPLAR_SIZE / s_z #后处理中使用
    d_search = (instance_size - EXEMPLAR_SIZE) / 2  # slightly different from rpn++
    pad = d_search / scale_z
    # s_x = python2round(s_z + 2 * pad)
    s_x = round(s_z + 2 * pad)
    x_crop = get_subwindow_tracking(image, target_pos, instance_size, s_x, avg_chans)
    return x_crop, scale_z

def numeric_sort_key(s):
    return int(re.search(r'\d+', s).group())
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def change(r):
    return np.maximum(r, 1. / r)
def sz(w,h):
    pad = (w + h) * 0.5
    sz2 = (w + pad) * (h + pad)
    return np.sqrt(sz2)

def postprocess(cls_score,bbox_pred,scale_z,target_pos,t_sz, grid_search_x, grid_search_y,hanning_window,instance_size):
    # 后处理
    pred_x1 = grid_search_x - bbox_pred[0, ...]
    pred_y1 = grid_search_y - bbox_pred[1, ...]
    pred_x2 = grid_search_x + bbox_pred[2, ...]
    pred_y2 = grid_search_y + bbox_pred[3, ...]       
    
    # size penalty
    s_c = change(sz(pred_x2 - pred_x1, pred_y2 - pred_y1) / (sz(t_sz[0],t_sz[1])))  # scale penalty
    r_c = change((t_sz[0] / t_sz[1]) / ((pred_x2 - pred_x1) / (pred_y2 - pred_y1)))  # ratio penalty

    penalty = np.exp(-(r_c * s_c - 1) * PENALTY_K)
    pscore = penalty * cls_score
    
    # window penalty
    pscore = pscore * (1 - WINDOW_INFLUENCE) + hanning_window * WINDOW_INFLUENCE
    
    # get max
    r_max, c_max = np.unravel_index(pscore.argmax(), pscore.shape)
    
    # to real size
    pred_x1 = pred_x1[r_max, c_max]
    pred_y1 = pred_y1[r_max, c_max]
    pred_x2 = pred_x2[r_max, c_max]
    pred_y2 = pred_y2[r_max, c_max]

    pred_xs = (pred_x1 + pred_x2) / 2
    pred_ys = (pred_y1 + pred_y2) / 2
    pred_w = pred_x2 - pred_x1
    pred_h = pred_y2 - pred_y1


    diff_xs = pred_xs - instance_size // 2
    diff_ys = pred_ys - instance_size // 2
    
    diff_xs, diff_ys, pred_w, pred_h = diff_xs / scale_z, diff_ys / scale_z, pred_w / scale_z, pred_h / scale_z

    new_t_sz = t_sz / scale_z

    # size learning rate
    plr = penalty[r_max, c_max] * cls_score[r_max, c_max] * LR


    # size rate
    res_xs = target_pos[0] + diff_xs
    res_ys = target_pos[1] + diff_ys
    res_w = pred_w * plr + (1 - plr) * new_t_sz[0]
    res_h = pred_h * plr + (1 - plr) * new_t_sz[1]

    target_pos = np.array([res_xs, res_ys])
    new_t_sz = new_t_sz * (1 - plr) + plr * np.array([res_w, res_h])
    return target_pos, new_t_sz, cls_score
def cxy_wh_2_rect(pos, sz):
    return [float(max(float(0), pos[0]-sz[0]/2)), float(max(float(0), pos[1]-sz[1]/2)), float(sz[0]), float(sz[1])]  # 0-index


def read_list_from_txt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]



