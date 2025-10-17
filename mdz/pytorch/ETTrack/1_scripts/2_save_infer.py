import os
import sys
import time
from collections import OrderedDict
import types

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from pytracking.evaluation import get_dataset
from pytracking.evaluation.running import _save_tracker_output
from pytracking.evaluation import Tracker

import torch
import numpy as np
from tracking.basic_model.et_tracker import ET_Tracker
from pytracking.tracker.et_tracker.et_tracker import TransconverTracker

def ET_Tracker_template(self, z):
    '''
    Used during the tracking -> computes the embedding of the target in the first frame.
    '''
    # self.zf = self.backbone_net(z)
    # 替换为加载trace出来的模型net1
    net1 = torch.jit.load(NET1_PATH)
    global zf
    zf = net1(z)#存为全局变量
def ET_Tracker_forward(self, x, zf):
    # [1,3,288,288]
    xf = self.backbone_net(x)
    # [1,96,16,16]
    # Batch Normalization before Corr
    # ICRAFT NOTE:
    # 为了部署，将成员变量作为前向输入
    # zf, xf = self.neck(self.zf, xf)  #[1,96,8,8] [1,96,16,16]<-[1,96,8,8] [1,96,16,16]
    zf, xf = self.neck(zf, xf)  #[1,96,8,8] [1,96,16,16]<-[1,96,8,8] [1,96,16,16]

    # ICRAFT NOTE:
    # 不支持字典传递数据，将feature_fusor返回数据展开
    # pixelwise correlation
    # feat_dict = self.feature_fusor(zf, xf) # cls:[1,128,16,16],[1,128,16,16]<-[1,96,8,8] [1,96,16,16]
    feat_cls, feat_reg = self.feature_fusor(zf, xf) # cls:[1,128,16,16],[1,128,16,16]<-[1,96,8,8] [1,96,16,16]

    c = self.cls_branch_1(feat_cls)#(feat_dict['cls'])
    c = self.cls_branch_2(c) 
    c = self.cls_branch_3(c) 
    c = self.cls_branch_4(c) 
    c = self.cls_branch_5(c) 
    c = self.cls_branch_6(c)
    c = self.cls_pred_head(c) # [1,1,16,16]
    
    b = self.bbreg_branch_1(feat_reg)#(feat_dict['reg'])
    b = self.bbreg_branch_2(b) 
    b = self.bbreg_branch_3(b) 
    b = self.bbreg_branch_4(b) 
    b = self.bbreg_branch_5(b) 
    b = self.bbreg_branch_6(b) 
    b = self.bbreg_branch_7(b) 
    b = self.bbreg_branch_8(b)
    b = self.reg_pred_head(b) # [1,4,16,16]
    # oup = {}
    # oup['cls'] = c
    # oup['reg'] = b
    # ICRAFT NOTE:
    # 不支持字典传递数据，直接返回
    # return oup['cls'], oup['reg']
    return c, b
ET_Tracker.template = ET_Tracker_template
ET_Tracker.forward = ET_Tracker_forward

def TransconverTracker_update(self, x_crops, target_pos, target_sz, window, scale_z, p, debug=False, writer=None):

    # cls_score, bbox_pred = self.net.track(x_crops.to(self.params.device))
    # 替换为加载trace出来的模型net2
    net2 = torch.jit.load(NET2_PATH)
    cls_score, bbox_pred = net2(x_crops.to(self.params.device),zf)

    cls_score = torch.sigmoid(cls_score).squeeze().cpu().data.numpy()

    # bbox to real predict
    bbox_pred = bbox_pred.squeeze().cpu().data.numpy()

    pred_x1 = self.grid_to_search_x - bbox_pred[0, ...]
    pred_y1 = self.grid_to_search_y - bbox_pred[1, ...]
    pred_x2 = self.grid_to_search_x + bbox_pred[2, ...]
    pred_y2 = self.grid_to_search_y + bbox_pred[3, ...]       
    

    # size penalty
    s_c = self.change(self.sz(pred_x2 - pred_x1, pred_y2 - pred_y1) / (self.sz_wh(target_sz)))  # scale penalty
    r_c = self.change((target_sz[0] / target_sz[1]) / ((pred_x2 - pred_x1) / (pred_y2 - pred_y1)))  # ratio penalty

    penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)
    pscore = penalty * cls_score
    
    # window penalty
    pscore = pscore * (1 - p.window_influence) + window * p.window_influence
    
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

    diff_xs = pred_xs - p.instance_size // 2
    diff_ys = pred_ys - p.instance_size // 2
    
    diff_xs, diff_ys, pred_w, pred_h = diff_xs / scale_z, diff_ys / scale_z, pred_w / scale_z, pred_h / scale_z

    target_sz = target_sz / scale_z

    # size learning rate
    lr = penalty[r_max, c_max] * cls_score[r_max, c_max] * p.lr

    # size rate
    res_xs = target_pos[0] + diff_xs
    res_ys = target_pos[1] + diff_ys
    res_w = pred_w * lr + (1 - lr) * target_sz[0]
    res_h = pred_h * lr + (1 - lr) * target_sz[1]

    target_pos = np.array([res_xs, res_ys])
    target_sz = target_sz * (1 - lr) + lr * np.array([res_w, res_h])

    if debug:
        return target_pos, target_sz, cls_score[r_max, c_max], cls_score
    else:
        return target_pos, target_sz, cls_score[r_max, c_max]
    
TransconverTracker.update =TransconverTracker_update

# 全局变量
TRACE_PATH = "../2_compile/fmodel/"
NET1_PATH = TRACE_PATH+"ettrack_net1_1x3x127x127_traced.pt"
NET2_PATH = TRACE_PATH+"ettrack_net2_1x3x288x288_traced.pt"

zf = torch.zeros(1,96,8,8)
if __name__ == '__main__':
    dataset_name = 'lasot'
    tracker_name = 'et_tracker'
    tracker_param = 'et_tracker'
    visualization=None
    debug=None
    visdom_info=None
    run_id = 2405101502
    dataset = get_dataset(dataset_name)

    tracker = Tracker(tracker_name, tracker_param, run_id)

    params = tracker.get_parameters()
    visualization_ = visualization

    

    debug_ = debug
    if debug is None:
        debug_ = getattr(params, 'debug', 0)
    if visualization is None:
        if debug is None:
            visualization_ = getattr(params, 'visualization', False)
        else:
            visualization_ = True if debug else False

    params.visualization = visualization_
    params.debug = debug_

    for seq in dataset[:]:
        print(seq)
        def _results_exist():
            if seq.dataset == 'oxuva':
                vid_id, obj_id = seq.name.split('_')[:2]
                pred_file = os.path.join(tracker.results_dir, '{}_{}.csv'.format(vid_id, obj_id))
                return os.path.isfile(pred_file)
            elif seq.object_ids is None:
                bbox_file = '{}/{}.txt'.format(tracker.results_dir, seq.name)
                return os.path.isfile(bbox_file)
            else:
                bbox_files = ['{}/{}_{}.txt'.format(tracker.results_dir, seq.name, obj_id) for obj_id in seq.object_ids]
                missing = [not os.path.isfile(f) for f in bbox_files]
                return sum(missing) == 0

        visdom_info = {} if visdom_info is None else visdom_info

        if _results_exist() and not debug:
            print('FPS: {}'.format(-1))
            continue

        print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq.name))

        tracker._init_visdom(visdom_info, debug_)
        if visualization_ and tracker.visdom is None:
            tracker.init_visualization()

        # Get init information
        init_info = seq.init_info()
        et_tracker = tracker.create_tracker(params)
        output = {'target_bbox': [],
            'time': [],
            'segmentation': [],
            'object_presence_score': []}

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = tracker._read_image(seq.frames[0])

        if et_tracker.params.visualization and tracker.visdom is None:
            tracker.visualize(image, init_info.get('init_bbox'))

        start_time = time.time()
        out = et_tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)

        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time,
                        'segmentation': init_info.get('init_mask'),
                        'object_presence_score': 1.}

        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = tracker._read_image(frame_path)

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            out = et_tracker.track(image, info)
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

            segmentation = out['segmentation'] if 'segmentation' in out else None
            if tracker.visdom is not None:
                tracker.visdom_draw_tracking(image, out['target_bbox'], segmentation)
            elif et_tracker.params.visualization:
                tracker.visualize(image, out['target_bbox'], segmentation)

        for key in ['target_bbox', 'segmentation']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        output['image_shape'] = image.shape[:2]
        output['object_presence_score_threshold'] = et_tracker.params.get('object_presence_score_threshold', 0.55)

        sys.stdout.flush()

        if isinstance(output['time'][0], (dict, OrderedDict)):
            exec_time = sum([sum(times.values()) for times in output['time']])
            num_frames = len(output['time'])
        else:
            exec_time = sum(output['time'])
            num_frames = len(output['time'])

        print('FPS: {}'.format(num_frames / exec_time))

        if not debug:
            _save_tracker_output(seq, tracker, output)