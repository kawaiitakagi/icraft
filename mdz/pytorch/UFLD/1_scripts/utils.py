
# postprocess
# post_process + get_lanes + to_array
import scipy
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np 
import os
import os.path as osp
import cv2
# set parameters
ORI_IMG_H = 720
ORI_IMG_W = 1280
IMG_H = 288
IMG_W = 800
CUT_HEIGHT = 0
GRIDING_NUM = 100
SAMPLE_Y = list(range(710, 150, -10))
INVALID_VALUE = -2
def preprocess(ori_img,IMG_W = 800,IMG_H = 288):
    # Resize img to (1,288,800,3)
    img = cv2.resize(ori_img,(IMG_W,IMG_H),interpolation=cv2.INTER_CUBIC)
    
    img = np.expand_dims(img,axis=0)
    
    return img 

def postprocess(out, localization_type='rel', flip_updown=True):
    predictions = []
    
    for j in range(out.shape[0]):
        # out_j = out[j].data.cpu().numpy()
        out_j = out[j]
        if flip_updown:
            out_j = out_j[:, ::-1, :]
        if localization_type == 'abs':
            out_j = np.argmax(out_j, axis=0)
            out_j[out_j == GRIDING_NUM] = -1
            out_j = out_j + 1
        elif localization_type == 'rel':
            prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
            idx = np.arange(GRIDING_NUM) + 1
            idx = idx.reshape(-1, 1, 1)
            loc = np.sum(prob * idx, axis=0)
            out_j = np.argmax(out_j, axis=0)
            loc[out_j == GRIDING_NUM] = 0
            out_j = loc
        else:
            raise NotImplementedError
        predictions.append(out_j)
    return predictions

def get_lanes(pred):
    '''
    1.postprocess
    2.calculate [x,y]
    3.[x,y] / img_w or img_h
    '''
    # postprocess
    predictions = postprocess(pred) 
    # get lanes
    ret = []
    for out in predictions:
        lanes = []
        # each lane
        for i in range(out.shape[1]):  
            if sum(out[:, i] != 0) <= 2: continue
            out_i = out[:, i]
            coord = []
            # each point
            for k in range(out.shape[0]):
                if out[k, i] <= 0: continue
                # calculate x,y point 
                x = ((out_i[k]-0.5) * ORI_IMG_W / (GRIDING_NUM - 1))
                y = SAMPLE_Y[k]
                coord.append([x, y])
            coord = np.array(coord)
            coord = np.flip(coord, axis=0)
            coord[:, 0] /= ORI_IMG_W
            coord[:, 1] /= ORI_IMG_H
            lanes.append(coord)
        ret.append(lanes)
    return lanes

# points = lane for lane in lanes
def to_array(points):
    '''
    lane to array 
    1.interpolate points
    2.[x,y] renormalize to ori_img_h and ori_img_w
    '''
    ys = np.array(SAMPLE_Y) / float(ORI_IMG_H)
    min_y = points[:, 1].min() - 0.01
    max_y = points[:, 1].max() + 0.01
    # function 
    function = InterpolatedUnivariateSpline(points[:, 1], points[:, 0], k=min(3, len(points) - 1))
    xs = function(ys)
    xs[(ys < min_y) | (ys > max_y)] = INVALID_VALUE
    
    valid_mask = (xs >= 0) & (xs < 1)
    lane_xs = xs[valid_mask] * ORI_IMG_W
    lane_ys = ys[valid_mask] * ORI_IMG_H
    lane = np.concatenate((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), axis=1)
    return lane

#=======================
# show lane results
#=======================
def imshow_lanes(img, lanes, out_file=None):
    for lane in lanes:
        for x, y in lane:
            if x <= 0 or y <= 0:
                continue
            x, y = int(x), int(y)
            cv2.circle(img, (x, y), 4, (255, 0, 0), 2)

    if out_file:
        if not osp.exists(osp.dirname(out_file)):
            os.makedirs(osp.dirname(out_file))
        cv2.imwrite(out_file, img)
       
        print('====Save results at:',out_file,'====')
