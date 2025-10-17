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
WEIGHTS_PATH = '../weights/maskformer_panoptic_swin_tiny_bs64_554k.pkl'
# IMG_PATH = '../2_compile/qtset/coco/000000001000.jpg'
IMG_PATH =  'E:\\Dataset\\coco\\val2017\\000000000139.jpg'
FIXED_H = 640
FIXED_W = 640
TRACED_MODEL_PATH = "../2_compile/fmodel/maskformer_pan_"+str(FIXED_H)+"x"+str(FIXED_W)+".onnx"



def panoptic_inference(mask_cls, mask_pred):
    scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
    mask_pred = mask_pred.sigmoid()

    keep = labels.ne(133) & (scores >0.8)
    cur_scores = scores[keep]
    cur_classes = labels[keep]
    cur_masks = mask_pred[keep]
    cur_mask_cls = mask_cls[keep]
    cur_mask_cls = cur_mask_cls[:, :-1]

    cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

    h, w = cur_masks.shape[-2:]
    panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
    segments_info = []

    current_segment_id = 0

    if cur_masks.shape[0] == 0:
        # We didn't detect any mask :(
        return panoptic_seg, segments_info
    else:
        # take argmax
        cur_mask_ids = cur_prob_masks.argmax(0)
        stuff_memory_list = {}
        for k in range(cur_classes.shape[0]):
            pred_class = cur_classes[k].item()
            isthing = pred_class in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                                      19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 
                                      36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53
                                      , 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
                                        72, 73, 74, 75, 76, 77, 78, 79]

            mask = cur_mask_ids == k
            mask_area = mask.sum().item()
            original_area = (cur_masks[k] >= 0.5).sum().item()

            if mask_area > 0 and original_area > 0:
                if mask_area / original_area < 0.8:
                    continue

                # merge stuff regions
                if not isthing:
                    if int(pred_class) in stuff_memory_list.keys():
                        panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                        continue
                    else:
                        stuff_memory_list[int(pred_class)] = current_segment_id + 1

                current_segment_id += 1
                panoptic_seg[mask] = current_segment_id

                segments_info.append(
                    {
                        "id": current_segment_id,
                        "isthing": bool(isthing),
                        "category_id": int(pred_class),
                    }
                )

        return panoptic_seg, segments_info
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
    processed_result = []
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
        mask_pred_result = sem_seg_postprocess(mask_pred_result, image.image_sizes[0], height, width)
        panoptic_r = panoptic_inference(mask_cls_result, mask_pred_result)
    metadata = MetadataCatalog.get('coco_2017_val_panoptic')
    visualizer = Visualizer(original_image, metadata, instance_mode=ColorMode.IMAGE)
    panoptic_seg, segments_info = panoptic_r
    vis_output = visualizer.draw_panoptic_seg_predictions(panoptic_seg, segments_info)
    # vis_output = visualizer.draw_sem_seg(r.argmax(dim=0))

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
