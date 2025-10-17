from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import argparse
import cv2
import torch
import numpy as np
from glob import glob
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)
from pysot.tracker.siamrpn_tracker import SiamRPNTracker
from pysot.models.model_builder import ModelBuilder

TRACE_MODEL_NAME = "../2_compile/fmodel/siamrpn++-net-2-255x255.pt"

def parse_arg():
    parser = argparse.ArgumentParser(description='tracking demo')
    parser.add_argument('--config', type=str, default="experiments/siamrpn_mobilev2_l234_dwxcorr/config.yaml",help='config file')
    parser.add_argument('--snapshot', type=str, default="experiments/siamrpn_mobilev2_l234_dwxcorr/model.pth", help='model name')
    parser.add_argument('--video_name', default='video.mp4', type=str,#./demo/bag.avi
                        help='videos or image files')
    args = parser.parse_args()
    return args

def new_forward(self, x,k1,k2,k3,k4,k5,k6):
    xf = self.backbone(x)
    xf = self.neck(xf)

    out = self.rpn_head.net2_forward(xf)
    x1,x2,x3,x4,x5,x6 = out[0],out[1],out[2],out[3],out[4],out[5]

    cls,loc = self.rpn_head(x1,x2,x3,x4,x5,x6,k1,k2,k3,k4,k5,k6)
    return cls,loc
ModelBuilder.forward = new_forward

def new_track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        outputs ={}
        k1 = torch.Tensor(np.fromfile(f'./k1.ftmp',np.float32).reshape(256,1,5,5))
        k2 = torch.Tensor(np.fromfile(f'./k2.ftmp',np.float32).reshape(256,1,5,5))
        k3 = torch.Tensor(np.fromfile(f'./k3.ftmp',np.float32).reshape(256,1,5,5))
        k4 = torch.Tensor(np.fromfile(f'./k4.ftmp',np.float32).reshape(256,1,5,5))
        k5 = torch.Tensor(np.fromfile(f'./k5.ftmp',np.float32).reshape(256,1,5,5))
        k6 = torch.Tensor(np.fromfile(f'./k6.ftmp',np.float32).reshape(256,1,5,5))
        outputs['cls'],outputs['loc'] = self.model(x_crop,k1,k2,k3,k4,k5,k6)
        # 导出模型siamrpn++-net2
        torch.jit.save(torch.jit.trace(self.model,(x_crop,k1,k2,k3,k4,k5,k6)),TRACE_MODEL_NAME)
        print("successful saved ",TRACE_MODEL_NAME)

        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        # print(self.window.shape)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        return {
                'bbox': bbox,
                'best_score': best_score
               }
SiamRPNTracker.track = new_track


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main():
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu()),False)
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)

    index = 0
    for frame in get_frames(args.video_name):
        if first_frame:
            try:
                # 获得追踪目标 xywh
                init_rect = cv2.selectROI(video_name, frame, False, False)
                # init_rect= (309, 377, 211, 114)
            except:
                exit()
            # print('init_rect.size=',init_rect)
            # init_rect= (309, 377, 211, 114)

            tracker.init(frame, init_rect)
            first_frame = False
        else:
            # pass
            outputs = tracker.track(frame)
            break
            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            else:
                bbox = list(map(int, outputs['bbox']))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (0, 255, 0), 3)
            cv2.imshow(video_name, frame)
            # cv2.imwrite(f"./track_result/{index}.jpg",frame)
            index += 1
            cv2.waitKey(1)


if __name__ == '__main__':
    # load config
    args = parse_arg()
    main()
