import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.append(R"../0_yolov9")
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (LOGGER, Profile, check_img_size, colorstr, cv2,
                           increment_path, non_max_suppression, scale_boxes)
from utils.plots import Annotator, colors
from utils.panoptic.general import process_mask
from utils.torch_utils import select_device, smart_inference_mode

@smart_inference_mode()
def run(
    weights= 'gelan-c-pan.pt',  # model.pt path(s)
    source= 'data/images',  # file/dir/URL/glob/screen/0(webcam)
    data= 'data/coco.yaml',  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    visualize=False,  # visualize features
    project= './runs/predict-pan',  # save results to project/name
    name='exp',  # save results to project/name
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
):
    source = str(source)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=False) # increment run
    save_dir.mkdir(parents=True, exist_ok=True) # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, data=data)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    # im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # longside resize shortside padded
    # im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    # im = np.ascontiguousarray(im)  # contiguous
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=False)

    # Run inference 
    seen, dt = 0, (Profile(), Profile(), Profile())
    for path, im, im0s, _, s in dataset:
        # prepare image
        with dt[0]:  
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]: 
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred, pred_out = model(im, augment=False, visualize=visualize)
            _, _, proto, psemasks = pred_out

        # NMS
        with dt[2]: 
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
        
        # Process semanic masks
        
        _, cls, h0, w0 = im.shape
        psemask = torch.nn.functional.interpolate(psemasks, size = (h0, w0), mode = 'bilinear', align_corners = False) 
        psemask = torch.squeeze(psemask) # shape: [CLASS, H , W], CLASS= 80 + 93 = 173
        semantic_mask = torch.flatten(psemask, start_dim = 1).permute(1, 0)  # class x h x w -> (h x w) x class
        max_idx = semantic_mask.argmax(1)
        unique_labels, inverse_indices = torch.unique(max_idx, return_inverse=True)  # shape: [N] 和 [H*W]
        inverse_indices = inverse_indices.view(h0, w0)
        N = unique_labels.size(0)
        output_masks = torch.zeros((N, h0, w0))
        output_masks = (inverse_indices.unsqueeze(0) == torch.arange(N).unsqueeze(1).unsqueeze(2)).byte()
        im_pan = im0s.copy()
        annotator_pan = Annotator(im_pan, line_width=line_thickness, example=str(names))
        annotator_pan.masks(output_masks,
                            colors=[colors(x, True) for x in unique_labels],
                            alpha=1)
        # Stream results
        im_pan = annotator_pan.result()
        save_path = str(save_dir / ('semantic_' + Path(path).name)) 
        cv2.imwrite(save_path, im_pan)

        # Process boxes & instance masks
        for i, det in enumerate(pred):    # per image
            seen += 1

            p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name) # im.jpg

            s += '%gx%g ' % im.shape[2:]  # print string

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))   

            if len(det):
                masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True) # HWC
                # print(masks.shape)
                # masks.shape: tensor [num of masks, h0, w0]
                # im0.shape: (h, w, c)
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                annotator.masks(masks,
                                colors=[colors(x, True) for x in det[:, 5]],
                                )

                # Write results
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

            # Stream results
            im0 = annotator.result()
            cv2.imwrite(save_path, im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default= '../weights/gelan-c-pan.pt', help='model path')
    parser.add_argument('--source', type=str, default= '../0_yolov9/data/images/horses.jpg', help='img path')
    opt = parser.parse_args()
    
    run(**vars(opt))
