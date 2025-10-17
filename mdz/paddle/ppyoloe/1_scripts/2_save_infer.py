# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
IMAGE_PATH = "../2_compile/qtset/coco/000000000139.jpg"
TRACED_MODEL_PATH = "../2_compile/fmodel/ppyoloe_plus_crn_s_80e_coco_640x640.onnx"
import os
import sys
import onnx

import onnxruntime as rt
# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

# ignore warning log
import warnings
warnings.filterwarnings('ignore')
import glob
import ast

import paddle
from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer
from ppdet.utils.check import check_gpu, check_npu, check_xpu, check_mlu, check_version, check_config
from ppdet.utils.cli import ArgsParser, merge_args
from ppdet.slim import build_slim_model
import paddle.nn.functional as F

from ppdet.utils.logger import setup_logger
logger = setup_logger('train')


from ppdet.modeling.architectures.yolo import YOLOv3
def my_forward(self):

    sess = rt.InferenceSession(TRACED_MODEL_PATH)
    input_name = sess.get_inputs()[0].name

    pred_onnx = sess.run(None, {input_name: self.inputs['image'].numpy()})

    if self.yolo_head.eval_size:
        anchor_points, stride_tensor = self.yolo_head.anchor_points, self.yolo_head.stride_tensor

    cls_score_list, reg_dist_list = [], []

    cls_logit_0 = paddle.to_tensor(pred_onnx[0],dtype='float32')
    cls_score = F.sigmoid(cls_logit_0)
    cls_score_list.append(cls_score.reshape([-1, self.yolo_head.num_classes, 400]))

    cls_logit_1 = paddle.to_tensor(pred_onnx[2],dtype='float32')
    cls_score = F.sigmoid(cls_logit_1)
    cls_score_list.append(cls_score.reshape([-1, self.yolo_head.num_classes, 1600]))

    cls_logit_2 = paddle.to_tensor(pred_onnx[4],dtype='float32')
    cls_score = F.sigmoid(cls_logit_2)
    cls_score_list.append(cls_score.reshape([-1, self.yolo_head.num_classes, 6400]))

    reg_dist_0 = paddle.to_tensor(pred_onnx[1],dtype='float32')
    reg_dist = reg_dist_0.reshape([-1, 4, self.yolo_head.reg_channels, 400]).transpose([0, 2, 3, 1])
    if self.yolo_head.use_shared_conv:
        reg_dist = self.yolo_head.proj_conv(F.softmax(
            reg_dist, axis=1)).squeeze(1)
    else:
        reg_dist = F.softmax(reg_dist, axis=1)
    reg_dist_list.append(reg_dist)

    reg_dist_1 = paddle.to_tensor(pred_onnx[3],dtype='float32')
    reg_dist = reg_dist_1.reshape([-1, 4, self.yolo_head.reg_channels, 1600]).transpose([0, 2, 3, 1])
    if self.yolo_head.use_shared_conv:
        reg_dist = self.yolo_head.proj_conv(F.softmax(
            reg_dist, axis=1)).squeeze(1)
    else:
        reg_dist = F.softmax(reg_dist, axis=1)
    reg_dist_list.append(reg_dist)

    reg_dist_2 = paddle.to_tensor(pred_onnx[5],dtype='float32')
    reg_dist = reg_dist_2.reshape([-1, 4, self.yolo_head.reg_channels, 6400]).transpose([0, 2, 3, 1])
    if self.yolo_head.use_shared_conv:
        reg_dist = self.yolo_head.proj_conv(F.softmax(
            reg_dist, axis=1)).squeeze(1)
    else:
        reg_dist = F.softmax(reg_dist, axis=1)
    reg_dist_list.append(reg_dist)


    cls_score_list = paddle.concat(cls_score_list, axis=-1)
    if self.yolo_head.use_shared_conv:
        reg_dist_list = paddle.concat(reg_dist_list, axis=1)
    else:
        reg_dist_list = paddle.concat(reg_dist_list, axis=2)
        reg_dist_list = self.yolo_head.proj_conv(reg_dist_list).squeeze(1)

    yolo_head_outs = [cls_score_list, reg_dist_list, anchor_points, stride_tensor]


    if self.for_mot:
        # the detection part of JDE MOT model
        boxes_idx, bbox, bbox_num, nms_keep_idx = self.post_process(
            yolo_head_outs, self.yolo_head.mask_anchors)
        output = {
            'bbox': bbox,
            'bbox_num': bbox_num,
            'boxes_idx': boxes_idx,
            'nms_keep_idx': nms_keep_idx,
            # 'emb_feats': emb_feats,
        }
    else:
        if self.return_idx:
            # the detection part of JDE MOT model
            _, bbox, bbox_num, nms_keep_idx = self.post_process(
                yolo_head_outs, self.yolo_head.mask_anchors)
        elif self.post_process is not None:
            # anchor based YOLOs: YOLOv3,PP-YOLO,PP-YOLOv2 use mask_anchors
            bbox, bbox_num, nms_keep_idx = self.post_process(
                yolo_head_outs, self.yolo_head.mask_anchors,
                self.inputs['im_shape'], self.inputs['scale_factor'])
        else:
            # anchor free YOLOs: PP-YOLOE, PP-YOLOE+
            bbox, bbox_num, nms_keep_idx = self.yolo_head.post_process(
                yolo_head_outs, self.inputs['scale_factor'])

        if self.use_extra_data:
            extra_data = {}  # record the bbox output before nms, such like scores and nms_keep_idx
            """extra_data:{
                        'scores': predict scores,
                        'nms_keep_idx': bbox index before nms,
                        }
            """
            extra_data['scores'] = yolo_head_outs[0]  # predict scores (probability)
            # Todo: get logits output
            extra_data['nms_keep_idx'] = nms_keep_idx
            # Todo support for mask_anchors yolo
            output = {'bbox': bbox, 'bbox_num': bbox_num, 'extra_data': extra_data}
        else:
            output = {'bbox': bbox, 'bbox_num': bbox_num}

    return output
YOLOv3._forward = my_forward

def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        "--infer_dir",
        type=str,
        default=None,
        help="Directory for images to perform inference on.")
    parser.add_argument(
        "--infer_img",
        type=str,
        default=IMAGE_PATH,
        help="Image path, has higher priority over --infer_dir")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/save_infer",
        help="Directory for storing the output visualization files.")
    parser.add_argument(
        "--draw_threshold",
        type=float,
        default=0.5,
        help="Threshold to reserve the result for visualization.")
    parser.add_argument(
        "--slim_config",
        default=None,
        type=str,
        help="Configuration file of slim method.")
    parser.add_argument(
        "--use_vdl",
        type=bool,
        default=False,
        help="Whether to record the data to VisualDL.")
    parser.add_argument(
        '--vdl_log_dir',
        type=str,
        default="vdl_log_dir/image",
        help='VisualDL logging directory for image.')
    parser.add_argument(
        "--save_results",
        type=bool,
        default=False,
        help="Whether to save inference results to output_dir.")
    parser.add_argument(
        "--slice_infer",
        action='store_true',
        help="Whether to slice the image and merge the inference results for small object detection."
    )
    parser.add_argument(
        '--slice_size',
        nargs='+',
        type=int,
        default=[640, 640],
        help="Height of the sliced image.")
    parser.add_argument(
        "--overlap_ratio",
        nargs='+',
        type=float,
        default=[0.25, 0.25],
        help="Overlap height ratio of the sliced image.")
    parser.add_argument(
        "--combine_method",
        type=str,
        default='nms',
        help="Combine method of the sliced images' detection results, choose in ['nms', 'nmm', 'concat']."
    )
    parser.add_argument(
        "--match_threshold",
        type=float,
        default=0.6,
        help="Combine method matching threshold.")
    parser.add_argument(
        "--match_metric",
        type=str,
        default='ios',
        help="Combine method matching metric, choose in ['iou', 'ios'].")
    parser.add_argument(
        "--visualize",
        type=ast.literal_eval,
        default=True,
        help="Whether to save visualize results to output_dir.")
    args = parser.parse_args()
    return args


def get_test_images(infer_dir, infer_img):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--infer_img or --infer_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]

    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)

    assert len(images) > 0, "no image found in {}".format(infer_dir)
    logger.info("Found {} inference images in total.".format(len(images)))

    return images


def run(FLAGS, cfg):
    # build trainer
    trainer = Trainer(cfg, mode='test')

    # load weights
    trainer.load_weights(cfg.weights)

    # get inference images
    images = get_test_images(FLAGS.infer_dir, FLAGS.infer_img)

    # inference
    if FLAGS.slice_infer:
        trainer.slice_predict(
            images,
            slice_size=FLAGS.slice_size,
            overlap_ratio=FLAGS.overlap_ratio,
            combine_method=FLAGS.combine_method,
            match_threshold=FLAGS.match_threshold,
            match_metric=FLAGS.match_metric,
            draw_threshold=FLAGS.draw_threshold,
            output_dir=FLAGS.output_dir,
            save_results=FLAGS.save_results,
            visualize=FLAGS.visualize)
    else:
        trainer.predict(
            images,
            draw_threshold=FLAGS.draw_threshold,
            output_dir=FLAGS.output_dir,
            save_results=FLAGS.save_results,
            visualize=FLAGS.visualize)


def main():
    FLAGS = parse_args()
    cfg = load_config(FLAGS.config)
    merge_args(cfg, FLAGS)
    merge_config(FLAGS.opt)

    # disable npu in config by default
    if 'use_npu' not in cfg:
        cfg.use_npu = False

    # disable xpu in config by default
    if 'use_xpu' not in cfg:
        cfg.use_xpu = False

    if 'use_gpu' not in cfg:
        cfg.use_gpu = False

    # disable mlu in config by default
    if 'use_mlu' not in cfg:
        cfg.use_mlu = False

    if cfg.use_gpu:
        place = paddle.set_device('gpu')
    elif cfg.use_npu:
        place = paddle.set_device('npu')
    elif cfg.use_xpu:
        place = paddle.set_device('xpu')
    elif cfg.use_mlu:
        place = paddle.set_device('mlu')
    else:
        place = paddle.set_device('cpu')

    if FLAGS.slim_config:
        cfg = build_slim_model(cfg, FLAGS.slim_config, mode='test')

    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_npu(cfg.use_npu)
    check_xpu(cfg.use_xpu)
    check_mlu(cfg.use_mlu)
    check_version()

    run(FLAGS, cfg)


if __name__ == '__main__':
    main()
