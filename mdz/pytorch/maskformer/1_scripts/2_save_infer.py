# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os
import onnxruntime
# fmt: off
import sys
sys.path.append(R"../0_MaskFormer")
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm
import torch
from torch.nn import functional as F
from fvcore.transforms.transform import (
    CropTransform,
    HFlipTransform,
    NoOpTransform,
    Transform,
    TransformList,
)
from icraft_models.imask_former_model import *
from detectron2.data.detection_utils import read_image

from detectron2.data.transforms.transform  import ResizeTransform
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.utils.visualizer import ColorMode, Visualizer
# constants
WINDOW_NAME = "MaskFormer demo"
WEIGHTS_PATH = '../weights/maskformer_R50_bs16_160k.pkl'
IMG_PATH = '../2_compile/qtset/ade/ADE_val_00000036.jpg'
FIXED_H = 640
FIXED_W = 640
TRACED_MODEL_PATH = "../2_compile/fmodel/maskformer_R50_"+str(FIXED_H)+"x"+str(FIXED_W)+".onnx"

# 固定图片尺寸
def get_transform_fix_size(self, image):
    h, w = image.shape[:2]
    if self.is_range:
        size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
    else:
        size = np.random.choice(self.short_edge_length)
    if size == 0:
        return NoOpTransform()

    # newh, neww = ResizeShortestEdge.get_output_shape(h, w, size, self.max_size)
    newh, neww = (FIXED_H,FIXED_W) # revised
    return ResizeTransform(h, w, newh, neww, self.interp)
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()



from detectron2.data.transforms.augmentation_impl import ResizeShortestEdge
ResizeShortestEdge.get_transform = get_transform_fix_size

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--input",
        nargs="+",
        default=IMG_PATH,
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    img = read_image(args.input, format="BGR")
    with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
        # if self.input_format == "RGB":
        original_image = img[:, :, ::-1]
        height, width = original_image.shape[:2]
        image = ResizeShortestEdge(short_edge_length=[512, 512], max_size=2048).get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        pixel_mean = torch.tensor([[[123.6750]],[[116.2800]],[[103.5300]]])
        pixel_std = torch.tensor([[[58.3950]],[[57.1200]],[[57.3750]]])

        image = image.to('cpu')
        image = ImageList.from_tensors([image], 32)

        image.tensor = (image.tensor - pixel_mean) / pixel_std
        model = onnxruntime.InferenceSession(TRACED_MODEL_PATH,providers=['CPUExecutionProvider'])
        inputs = {model.get_inputs()[0].name: to_numpy(image.tensor)}
        mask_cls_results,mask_pred_results  = model.run(None, inputs)
        mask_cls_result = torch.Tensor(np.array(mask_cls_results))[0]
        mask_pred_result = torch.Tensor(np.array(mask_pred_results))[0]

        # semantic segmentation inference
        mask_cls = F.softmax(mask_cls_result, dim=-1)[..., :-1]
        mask_pred = mask_pred_result.sigmoid()
        r = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        r = sem_seg_postprocess(r, image.image_sizes[0], height, width)

    metadata = MetadataCatalog.get('ade20k_sem_seg_val' if len(('ade20k_sem_seg_val',)) else "__unused")
    visualizer = Visualizer(original_image, metadata, instance_mode=ColorMode.IMAGE)
    vis_output = visualizer.draw_sem_seg(r.argmax(dim=0))

    if args.output:
        if os.path.isdir(args.output):
            assert os.path.isdir(args.output), args.output
            out_filename = os.path.join(args.output,'2_save_infer_'+os.path.basename(args.input))
        else:
            os.mkdir(args.output)
            out_filename = os.path.join(args.output,'2_save_infer_'+os.path.basename(args.input))
        vis_output.save(out_filename)
        print("The output is saved in: ",out_filename)
    else:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.imshow(WINDOW_NAME, vis_output.get_image()[:, :, ::-1])
        cv2.waitKey(0)
