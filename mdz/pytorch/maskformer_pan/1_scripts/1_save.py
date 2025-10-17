# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

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
from fvcore.transforms.transform import NoOpTransform
from icraft_models.imask_former_model import *
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from projects.DeepLab.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from mask_former import add_mask_former_config
from demo.predictor import VisualizationDemo
from detectron2.data.transforms.transform  import ResizeTransform
from detectron2.modeling.postprocessing import sem_seg_postprocess

# constants
WINDOW_NAME = "MaskFormer demo"
WEIGHTS_PATH = '../weights/maskformer_panoptic_swin_tiny_bs64_554k.pkl'
IMG_PATH = '../2_compile/qtset/coco/000000001000.jpg'
FIXED_H = 640
FIXED_W = 640
TRACED_MODEL_PATH = "../2_compile/fmodel/maskformer_pan_"+str(FIXED_H)+"x"+str(FIXED_W)+".onnx"


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

def icraft_defaults_call(self, original_image):

    with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
        # Apply pre-processing to image.
        if self.input_format == "RGB":
            # whether the model expects BGR inputs or RGB
            original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        image = self.aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        # inputs = {"image": image, "height": height, "width": width}
        image = image.to(self.model.device)
        image = (image - self.model.pixel_mean) / self.model.pixel_std 
        # image = image.unsqueeze(0)
        image = ImageList.from_tensors([image], self.model.size_divisibility)
        # tgt = torch.zeros([100,1,256]).to(self.model.device)
        # pos_embedding = self.model.sem_seg_head.predictor.pe_layer(image,torch.zeros((1, 20, 20), device=image.device, dtype=torch.bool)) # fixed
        # traced_model = torch.jit.trace(self.model.cpu(),image)
        # traced_model.save("../2_compile/fmodel/maskformer_R50_bs16_160k_640_640_1in.pt")



        # mask_cls_results,mask_embed,mask_features = self.model(**dummy_inputs)
        mask_cls_results,mask_pred_results = self.model(image.tensor)
        
        # mask_pred_results = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        torch.onnx.export(self.model.cpu(),image.tensor,TRACED_MODEL_PATH,opset_version=17)
        print("Trace Done ! Traced model is saved to "+TRACED_MODEL_PATH)
        # mask_cls_results = outputs["pred_logits"]
        # mask_pred_results = outputs["pred_masks"]
        # upsample masks
        # mask_pred_results = F.interpolate(
        #     mask_pred_results,
        #     size=(image.tensor.shape[-2], image.tensor.shape[-1]),
        #     mode="bilinear",
        #     align_corners=False,
        # )

        processed_results = []
        for mask_cls_result, mask_pred_result in zip(
            mask_cls_results, mask_pred_results
        ):
            # height, width = original_image.shape[:2]

            if self.model.sem_seg_postprocess_before_inference:
                mask_pred_result = sem_seg_postprocess(
                    mask_pred_result, image.image_sizes[0], height, width
                )

            # semantic segmentation inference
            r = self.model.semantic_inference(mask_cls_result, mask_pred_result)
            if not self.model.sem_seg_postprocess_before_inference:
                r = sem_seg_postprocess(r, image.image_sizes[0], height, width)
            processed_results.append({"sem_seg": r})

            # panoptic segmentation inference
            if self.model.panoptic_on:
                panoptic_r = self.model.panoptic_inference(mask_cls_result, mask_pred_result)
                processed_results[-1]["panoptic_seg"] = panoptic_r

        return processed_results[0]
    
def icraft_MaskFormer_forward(self, inputs,pos,tgt):
    features = self.backbone(inputs)
    outputs = self.sem_seg_head(features,pos,tgt)
    return outputs

def icraft_MaskFormerHead_forward(self, features,pos,tgt):
    return self.layers(features,pos,tgt)
    
def icraft_layers(self, features,pos,tgt):
    mask_features, transformer_encoder_features = self.pixel_decoder.forward_features(features)
    if self.transformer_in_feature == "transformer_encoder":
        assert (
            transformer_encoder_features is not None
        ), "Please use the TransformerEncoderPixelDecoder."
        predictions = self.predictor(transformer_encoder_features, mask_features,pos,tgt)
    else:
        predictions = self.predictor(features[self.transformer_in_feature], mask_features,pos,tgt)
    return predictions

def icraft_TransformerPredictor_forward(self, x, mask_features,pos,tgt):
    # pos = self.pe_layer(x)

    src = x
    mask = None
    hs, memory = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos,tgt)

    if self.mask_classification:
        outputs_class = self.class_embed(hs)
        # out = {"pred_logits": outputs_class[-1]}
    else:
        out = {}


    mask_embed = self.mask_embed(hs[-1])
    mask_pred_results = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
    mask_pred_results = F.interpolate(
        mask_pred_results,
        size=(x.shape[-2]*32, x.shape[-1]*32),
        mode="bilinear",
        align_corners=False,
    )
        # outputs_seg_masks = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        # out["pred_masks"] = outputs_seg_masks
    # return outputs_class[-1],mask_embed,mask_features
    return outputs_class[-1],mask_pred_results

def icraft_Transformer_forward(self, src, mask, query_embed, pos_embed,tgt):
    # flatten NxCxHxW to HWxNxC
    bs, c, h, w = src.shape
    src = src.flatten(2).permute(2, 0, 1)
    # pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
    # query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1) # icraft bs为1 
    query_embed = query_embed.unsqueeze(1)
    if mask is not None:
        mask = mask.flatten(1)

    # tgt = torch.zeros_like(query_embed)
    memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
    hs = self.decoder(
        tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed
    )
    return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


from detectron2.data.transforms.augmentation_impl import ResizeShortestEdge
ResizeShortestEdge.get_transform = get_transform_fix_size

from detectron2.engine.defaults import DefaultPredictor
DefaultPredictor.__call__ = icraft_defaults_call

from mask_former.modeling.heads.mask_former_head import MaskFormerHead
MaskFormerHead.__call__ = icraft_MaskFormerHead_forward
MaskFormerHead.layers = icraft_layers

from mask_former.modeling.transformer.transformer_predictor import TransformerPredictor
TransformerPredictor.__call__ = icraft_TransformerPredictor_forward

from mask_former.modeling.transformer.transformer import Transformer
Transformer.__call__ = icraft_Transformer_forward


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_mask_former_config(cfg)
    cfg.MODEL.BACKBONE.IM_SIZE = [FIXED_H,FIXED_W]  #fix

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="../0_MaskFormer/configs/coco-panoptic/swin/maskformer_panoptic_swin_tiny_bs64_554k.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
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

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    cfg['MODEL']['DEVICE'] = 'cpu'
    cfg['MODEL']['WEIGHTS'] = WEIGHTS_PATH
    cfg['MODEL']['META_ARCHITECTURE'] = "MaskFormer_icraft"
    demo = VisualizationDemo(cfg)
    print("device:",demo.predictor.model.device)
    # demo.predictor.model.sem_seg_head.predictor.query_embed.weight = demo.predictor.model.sem_seg_head.predictor.query_embed.weight.unsqueeze(1)
    # demo.predictor.model.sem_seg_head.predictor.query_embed2 = demo.predictor.model.sem_seg_head.predictor.query_embed.weight.unsqueeze(1)
    # demo.predictor.model.sem_seg_head.predictor.query_embed.weight.unsqueeze(1).detach().requires_grad_(requires_grad=False)
    # demo.predictor.model.sem_seg_head.predictor.query_embed.weight.unsqueeze(1).detach().requires_grad_(False)
    # print(demo.predictor.model.sem_seg_head.predictor.query_embed.weight.unsqueeze(1).requires_grad)
    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        path = args.input
        # for path in tqdm.tqdm(args.input, disable=not args.output):
        #     # use PIL, to be consistent with evaluation
        img = read_image(path, format="BGR")
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)
        logger.info(
            "{}: {} in {:.2f}s".format(
                path,
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )

        if args.output:
            if os.path.isdir(args.output):
                assert os.path.isdir(args.output), args.output
                out_filename = os.path.join(args.output,'1_save_'+os.path.basename(path))
            else:
                os.mkdir(args.output)
                out_filename = os.path.join(args.output,'1_save_'+os.path.basename(path))
            visualized_output.save(out_filename)
            print("The output is saved in: ",out_filename)
        else:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
            # if cv2.waitKey(0) == 27:
            #     break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
