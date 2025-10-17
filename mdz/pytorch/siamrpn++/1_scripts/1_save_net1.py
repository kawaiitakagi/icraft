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

TRACE_MODEL_NAME = "../2_compile/fmodel/siamrpn++-net-1-127x127.pt"

def parse_arg():
    parser = argparse.ArgumentParser(description='tracking demo')
    parser.add_argument('--config', type=str, default="experiments/siamrpn_mobilev2_l234_dwxcorr/config.yaml",help='config file')
    parser.add_argument('--snapshot', type=str, default="experiments/siamrpn_mobilev2_l234_dwxcorr/model.pth", help='model name')
    parser.add_argument('--video_name', default='video.mp4', type=str,#./demo/bag.avi
                        help='videos or image files')
    args = parser.parse_args()
    return args

def new_init(self, img, bbox):
    """
    args:
        img(np.ndarray): BGR image
        bbox: (x, y, w, h) bbox
    """
    self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                bbox[1]+(bbox[3]-1)/2])  
    self.size = np.array([bbox[2], bbox[3]])
    # calculate z crop size
    w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
    h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
    s_z = round(np.sqrt(w_z * h_z))
    # calculate channle average
    # https://blog.csdn.net/jndingxin/article/details/112618714 求多通道均值
    self.channel_average = np.mean(img, axis=(0, 1))
    self.channel_average = [0.0, 0.0, 0.0]
    # get crop
    z_crop = self.get_subwindow(img, self.center_pos,
                                cfg.TRACK.EXEMPLAR_SIZE,
                                s_z, self.channel_average)
    # 导出模型siamrpn++-net1
    torch.jit.save(torch.jit.trace(self.model,z_crop),TRACE_MODEL_NAME)
    print("successful saved ",TRACE_MODEL_NAME)
SiamRPNTracker.init = new_init

def siamrpn_net1_forward(self, z):
    # icraft code
    zf = self.backbone(z)
    zf = self.neck(zf)
    zf = self.rpn_head.net1_forward(zf)
    k1,k2,k3,k4,k5,k6 = zf[0].view(256, 1, 5, 5),zf[1].view(256, 1, 5, 5),zf[2].view(256, 1, 5, 5),zf[3].view(256, 1, 5, 5),zf[4].view(256, 1, 5, 5),zf[5].view(256, 1, 5, 5)
    k1.detach().numpy().astype(np.float32).tofile('k1.ftmp')
    k2.detach().numpy().astype(np.float32).tofile('k2.ftmp')
    k3.detach().numpy().astype(np.float32).tofile('k3.ftmp')
    k4.detach().numpy().astype(np.float32).tofile('k4.ftmp')
    k5.detach().numpy().astype(np.float32).tofile('k5.ftmp')
    k6.detach().numpy().astype(np.float32).tofile('k6.ftmp')
    return k1,k2,k3,k4,k5,k6
ModelBuilder.forward = siamrpn_net1_forward




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
            break
        else:
            # pass
            outputs = tracker.track(frame)
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
