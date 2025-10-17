#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import sys
sys.path.append(R"../0_yolox")
import argparse
import os
from loguru import logger
import torch
from yolox.exp import get_exp
from yolox.models.network_blocks import Focus
from yolox.models.yolo_head import YOLOXHead
import torch
RED = '\033[31m'       # 设置前景色为红色
RESET = '\033[0m'      # 重置所有属性到默认值
ver = torch.__version__
assert ("1.6" in ver) or ("1.9" in ver), f"{RED}Unsupported PyTorch version: {ver}{RESET}"

def parser():
    parser = argparse.ArgumentParser("YOLOX torchscript deploy")
    parser.add_argument(
        "--output-name", type=str, default="yolox.torchscript.pt", help="output name of models"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default="experiment-name")
    parser.add_argument("-n", "--name", type=str, default="yolox_s", help="model name")
    parser.add_argument("-c", "--ckpt", default="../weights/yolox_s.pth", type=str, help="ckpt path")
    parser.add_argument('--export_dir', type=str, default=R'../2_compile/fmodel', help='weights path')
    parser.add_argument("--imgsz", nargs='+', type=int, default=[640], help="test img size")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser


FOCUS_WEIGHT = torch.zeros(12,3,2,2)
for i in range(12):
    j=i%3
    k=(i//3)%2
    l=0 if i<6 else 1
    FOCUS_WEIGHT[i,j,k,l]=1
FOCUS_BIAS = torch.zeros(12)

def focus_forward(self, x):
    # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
    x = torch.nn.functional.conv2d(x,FOCUS_WEIGHT,bias=FOCUS_BIAS,stride=2,padding=0,dilation=1,groups=1)
    return self.conv(x)

def yolo_head_forward(self, xin, labels=None, imgs=None):
    outputs = []
    outputs_ = []
    origin_preds = []
    x_shifts = []
    y_shifts = []
    expanded_strides = []

    for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
        zip(self.cls_convs, self.reg_convs, self.strides, xin)
    ):
        x = self.stems[k](x)
        cls_x = x
        reg_x = x

        reg_feat = reg_conv(reg_x)
        obj_output = self.obj_preds[k](reg_feat)
        reg_output = self.reg_preds[k](reg_feat)

        cls_feat = cls_conv(cls_x)
        cls_output = self.cls_preds[k](cls_feat)

        if self.training:
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            output, grid = self.get_output_and_grid(
                output, k, stride_this_level, xin[0].type()
            )
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(
                torch.zeros(1, grid.shape[1])
                .fill_(stride_this_level)
                .type_as(xin[0])
            )
            if self.use_l1:
                batch_size = reg_output.shape[0]
                hsize, wsize = reg_output.shape[-2:]
                reg_output = reg_output.view(
                    batch_size, self.n_anchors, 4, hsize, wsize
                )
                reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                    batch_size, -1, 4
                )
                origin_preds.append(reg_output.clone())

        else:
            output = torch.cat(
                [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
            )
            outputs_.append(obj_output)
            outputs_.append(reg_output)
            outputs_.append(cls_output)

        outputs.append(output)
        
    if self.training:
        return self.get_losses(
            imgs,
            x_shifts,
            y_shifts,
            expanded_strides,
            labels,
            torch.cat(outputs, 1),
            origin_preds,
            dtype=xin[0].dtype,
        )
    else:
        self.hw = [x.shape[-2:] for x in outputs]
        # [batch, n_anchors_all, 85]
        outputs = torch.cat(
            [x.flatten(start_dim=2) for x in outputs], dim=2
        ).permute(0, 2, 1)
        self.decode_in_inference = False
        if self.decode_in_inference:
            return self.decode_outputs(outputs, dtype=xin[0].type())
        else:
            return outputs_
        

@logger.catch
def main():
    args = parser().parse_args()
    logger.info("args value: {}".format(args))
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    args.imgsz *= 2 if len(args.imgsz) == 1 else 1  # expand
    exp.test_size = tuple(args.imgsz)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model()
    if args.ckpt is None:
        file_name = os.path.join(exp.output_dir, args.experiment_name)
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt

    # load the model state dict
    ckpt = torch.load(ckpt_file, map_location="cpu")

    model.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model.head.decode_in_inference = False

    logger.info("loading checkpoint done.")
    dummy_input = torch.randn(args.batch_size, 3, exp.test_size[0], exp.test_size[1])
    # 修改模型
    Focus.__call__ = focus_forward
    YOLOXHead.__call__ = yolo_head_forward

    mod = torch.jit.trace(model, dummy_input)

    if not os.path.exists(args.export_dir):
        os.makedirs(args.export_dir)

    f = args.export_dir + '\\' + args.name + f'_{exp.test_size[0]}x{exp.test_size[1]}_traced.pt'
    mod.save(f)
    logger.info("generated torchscript model named {}".format(f))


if __name__ == "__main__":
    main()
