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

import os
import sys

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

# ignore warning log
import warnings
warnings.filterwarnings('ignore')

import paddle
RED = '\033[31m'       # 设置前景色为红色
RESET = '\033[0m'      # 重置所有属性到默认值
ver = paddle.__version__
print(ver)
assert ("2.4.1" in ver) , f"{RED}Unsupported PyTorch version: {ver}{RESET}"
from ppdet.core.workspace import load_config, merge_config
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.utils.cli import ArgsParser
from ppdet.engine import Trainer
from ppdet.slim import build_slim_model
import paddle.nn.functional as F
from ppdet.utils.logger import setup_logger
logger = setup_logger('export_model')

# 固定模型输入大小
from ppdet.engine import Trainer
from ppdet.utils.fuse_utils import fuse_conv_bn
from ppdet.engine.export_utils import _dump_infer_config, _prune_input_spec
from paddle.static import InputSpec
MOT_ARCH = ['JDE', 'FairMOT', 'DeepSORT', 'ByteTrack', 'CenterTrack']
def fix_shape_get_infer_cfg_and_input_spec(self,
                                    save_dir,
                                    prune_input=True,
                                    kl_quant=False):
    image_shape = None
    im_shape = [None, 2]
    scale_factor = [None, 2]
    if self.cfg.architecture in MOT_ARCH:
        test_reader_name = 'TestMOTReader'
    else:
        test_reader_name = 'TestReader'
    if 'inputs_def' in self.cfg[test_reader_name]:
        inputs_def = self.cfg[test_reader_name]['inputs_def']
        image_shape = inputs_def.get('image_shape', None)
    # set image_shape=[None, 3, -1, -1] as default
    if image_shape is None:
        image_shape = [None, 3, -1, -1]

    if len(image_shape) == 3:
        image_shape = [None] + image_shape
    else:
        im_shape = [image_shape[0], 2]
        scale_factor = [image_shape[0], 2]

    if hasattr(self.model, 'deploy'):
        self.model.deploy = True

    if 'slim' not in self.cfg:
        for layer in self.model.sublayers():
            if hasattr(layer, 'convert_to_deploy'):
                layer.convert_to_deploy()

    if hasattr(self.cfg, 'export') and 'fuse_conv_bn' in self.cfg[
            'export'] and self.cfg['export']['fuse_conv_bn']:
        self.model = fuse_conv_bn(self.model)

    export_post_process = self.cfg['export'].get(
        'post_process', False) if hasattr(self.cfg, 'export') else True
    export_nms = self.cfg['export'].get('nms', False) if hasattr(
        self.cfg, 'export') else True
    export_benchmark = self.cfg['export'].get(
        'benchmark', False) if hasattr(self.cfg, 'export') else False
    if hasattr(self.model, 'fuse_norm'):
        self.model.fuse_norm = self.cfg['TestReader'].get('fuse_normalize',
                                                            False)
    if hasattr(self.model, 'export_post_process'):
        self.model.export_post_process = export_post_process if not export_benchmark else False
    if hasattr(self.model, 'export_nms'):
        self.model.export_nms = export_nms if not export_benchmark else False
    if export_post_process and not export_benchmark:
        image_shape = [None] + image_shape[1:]

    # Save infer cfg
    _dump_infer_config(self.cfg,
                        os.path.join(save_dir, 'infer_cfg.yml'), image_shape,
                        self.model)
    # print(image_shape)
    # print(scale_factor)
    # print(im_shape)
    image_shape = [1,3,640,640]
    scale_factor = [1,2]
    im_shape = [1,2]
    print("image_shape:",image_shape)
    print("scale_factor:",scale_factor)
    print("im_shape:",im_shape)
    input_spec = [{
        "image": InputSpec(
            shape=image_shape, name='image'),
        "im_shape": InputSpec(
            shape=im_shape, name='im_shape'),
        "scale_factor": InputSpec(
            shape=scale_factor, name='scale_factor')
    }]

    if self.cfg.architecture == 'DeepSORT':
        input_spec[0].update({
            "crops": InputSpec(
                shape=[None, 3, 192, 64], name='crops')
        })
    if prune_input:
        static_model = paddle.jit.to_static(
            self.model, input_spec=input_spec)
        # NOTE: dy2st do not pruned program, but jit.save will prune program
        # input spec, prune input spec here and save with pruned input spec
        pruned_input_spec = _prune_input_spec(
            input_spec, static_model.forward.main_program,
            static_model.forward.outputs)
    else:
        static_model = None
        pruned_input_spec = input_spec

    # TODO: Hard code, delete it when support prune input_spec.
    if self.cfg.architecture == 'PicoDet' and not export_post_process:
        pruned_input_spec = [{
            "image": InputSpec(
                shape=image_shape, name='image')
        }]
    if kl_quant:
        if self.cfg.architecture == 'PicoDet' or 'ppyoloe' in self.cfg.weights:
            pruned_input_spec = [{
                "image": InputSpec(
                    shape=image_shape, name='image'),
                "scale_factor": InputSpec(
                    shape=scale_factor, name='scale_factor')
            }]
        elif 'tinypose' in self.cfg.weights:
            pruned_input_spec = [{
                "image": InputSpec(
                    shape=image_shape, name='image')
            }]

    return static_model, pruned_input_spec
Trainer._get_infer_cfg_and_input_spec = fix_shape_get_infer_cfg_and_input_spec
# 解耦分类头和检测头
from ppdet.modeling.heads.ppyoloe_head import PPYOLOEHead
def my_forward_eval(self, feats):
    cls_dec_list = []
    for i, feat in enumerate(feats):
        b, _, h, w = feat.shape
        l = h * w
        avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                        feat)
        reg_dist = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))

        cls_dec_list.append(cls_logit)
        cls_dec_list.append(reg_dist)
    return cls_dec_list
PPYOLOEHead.forward_eval=my_forward_eval
from ppdet.modeling.architectures.yolo import YOLOv3
def my_forward(self):
    body_feats = self.backbone(self.inputs)
    neck_feats = self.neck(body_feats, self.for_mot)

    if isinstance(neck_feats, dict):
        assert self.for_mot == True
        emb_feats = neck_feats['emb_feats']
        neck_feats = neck_feats['yolo_feats']
        
    yolo_head_outs = self.yolo_head(neck_feats)

    return yolo_head_outs 
YOLOv3._forward = my_forward

def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_inference",
        help="Directory for storing the output model files.")
    parser.add_argument(
        "--export_serving_model",
        type=bool,
        default=False,
        help="Whether to export serving model or not.")
    parser.add_argument(
        "--slim_config",
        default=None,
        type=str,
        help="Configuration file of slim method.")
    args = parser.parse_args()
    return args


def run(FLAGS, cfg):
    # build detector
    trainer = Trainer(cfg, mode='test')

    # load weights
    if cfg.architecture in ['DeepSORT', 'ByteTrack']:
        trainer.load_weights_sde(cfg.det_weights, cfg.reid_weights)
    else:
        trainer.load_weights(cfg.weights)

    # export model
    trainer.export(FLAGS.output_dir)

    if FLAGS.export_serving_model:
        from paddle_serving_client.io import inference_model_to_serving
        model_name = os.path.splitext(os.path.split(cfg.filename)[-1])[0]

        inference_model_to_serving(
            dirname="{}/{}".format(FLAGS.output_dir, model_name),
            serving_server="{}/{}/serving_server".format(FLAGS.output_dir,
                                                         model_name),
            serving_client="{}/{}/serving_client".format(FLAGS.output_dir,
                                                         model_name),
            model_filename="model.pdmodel",
            params_filename="model.pdiparams")


def main():
    paddle.set_device("cpu")
    FLAGS = parse_args()
    cfg = load_config(FLAGS.config)
    merge_config(FLAGS.opt)

    if FLAGS.slim_config:
        cfg = build_slim_model(cfg, FLAGS.slim_config, mode='test')

    # FIXME: Temporarily solve the priority problem of FLAGS.opt
    merge_config(FLAGS.opt)
    check_config(cfg)
    if 'use_gpu' not in cfg:
        cfg.use_gpu = False
    check_gpu(cfg.use_gpu)
    check_version()

    run(FLAGS, cfg)


if __name__ == '__main__':
    main()
