from ast import Continue
import os
import sys
import argparse
import cv2
from matplotlib import category
import torch
import numpy as np
import json
sys.path.append(R"../0_yolov9")
from utils.tal.anchor_generator import make_anchors,dist2bbox
from models.common import DFL
from utils.panoptic.general import process_mask
from utils.general import scale_boxes, non_max_suppression
from utils.augmentations import letterbox
from utils.coco_utils import getCocoIds, getMappingId, getMappingIndex
from utils.panoptic.general import scale_image
from utils.plots import Annotator, colors
from pycocotools import mask as maskUtils
from yolov9_utils import panoptic_merge_show, crop_image#, non_max_suppression

COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
)

stuff_names = (
    "banner", "blanket", "branch", "bridge", "building-other", "bush", "cabinet", "cage", "cardboard",
    "carpet", "ceiling-other", "ceiling-tile", "cloth", "clothes", "clouds", "counter", "cupboard",
    "curtain", "desk-stuff", "dirt", "door-stuff", "fence", "floor-marble", "floor-other", "floor-stone",
    "floor-tile", "floor-wood", "flower", "fog", "food-other", "fruit", "furniture-other", "grass", "gravel",
    "ground-other", "hill", "house", "leaves", "light", "mat", "metal", "mirror-stuff", "moss", "mountain",
    "mud", "napkin", "net", "paper", "pavement", "pillow", "plant-other", "plastic", "platform", "playingfield",
    "railing", "railroad", "river", "road", "rock", "roof", "rug", "salad", "sand", "sea", "shelf", "sky-other",
    "skyscraper", "snow", "solid-other", "stairs", "stone", "straw", "structural-other", "table", "tent",
    "textile-other", "towel", "tree", "vegetable", "wall-brick", "wall-concrete", "wall-other", "wall-panel",
    "wall-stone", "wall-tile", "wall-wood", "water-other", "waterdrops", "window-blind", "window-other", "wood",
    "other", "unlabeled"
)

def pred_one_image(img_path, model_path):

    # preprocessing
    img_raw = cv2.imread(img_path)
    img_resize = letterbox(img_raw, 640, stride=32, auto=False)[0]  # padded resize
    im = img_resize.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im).float().unsqueeze(0)
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    # load traced model
    model = torch.jit.load(model_path)
    output = model(im)

    proto = output[9] # [1 , 32, 160, 160]

    psemasks = output[10] # [1, 80 + 93, 160, 160]

    # reunion-detect
    outputs_n1 = torch.cat((output[1], output[0]), 1) # [1, 64 + 80 , 80, 80] 
    outputs_n2 = torch.cat((output[4], output[3]), 1) # [1, 64 + 80 , 40, 40] 
    outputs_n3 = torch.cat((output[7], output[6]), 1) # [1, 64 + 80 , 20, 20] 
    outputs = []
    outputs.append(outputs_n1)
    outputs.append(outputs_n2)
    outputs.append(outputs_n3)

    # postprocess
    reg_max = 16
    nc = 80
    dfl_layer = DFL(reg_max)
    anchors, strides = (x.transpose(0, 1) for x in make_anchors(outputs, torch.from_numpy(np.array([8, 16, 32],dtype=np.float32)), 0.5))
    box, cls = torch.cat([xi.view(output[1].shape[0], reg_max*4+80, -1) for xi in outputs], 2).split((reg_max*4, nc), 1)

    dbox = dist2bbox(dfl_layer(box), anchors.unsqueeze(0), xywh=True, dim=1) * strides
    y = torch.cat((dbox, cls.sigmoid()), 1)
    # reunion-seg
    seg = torch.cat([output[2].view(1,32,-1), output[5].view(1,32,-1), output[8].view(1,32,-1)],2) # 80*80 + 40*40 + 20*20

    # final pred
    out = torch.cat((y,seg),1)   # [1, 84+32, 8400]

    # NMS
    conf_thres = 0.25
    iou_thres = 0.45
    
    pred = non_max_suppression(out, conf_thres, iou_thres, classes=None, agnostic=False, max_det=1000, nm=32)
    # pred = non_max_suppression(out.numpy(), conf_thres, iou_thres, classes=None, max_det=1000, nm=32)
    # print(pred[0][:, 5])
    # print(len(pred[0][:, 5]))

    if(pred[0][:, 6:].shape[0] == 0):
        pred[0] = torch.zeros(1,38)
        print("Cannot detec any instance masks!")
    masks = process_mask(proto[0], pred[0][:, 6:], pred[0][:, :4], im.shape[2:], upsample=True) # HWC
    pred[0][:, :4] = scale_boxes(im.shape[2:], pred[0][:, :4], img_raw.shape).round()  # rescale boxes to im0 size
    

    # process semask
    _, _, h0, w0 = im.shape
    im_pan = img_raw.copy()
    psemask = torch.nn.functional.interpolate(psemasks, size = (h0, w0), mode = 'bilinear', align_corners = False) 
    psemask = torch.squeeze(psemask) # shape: [CLASS, H , W], CLASS= 80 + 93 = 173
    nc, h, w = psemask.shape
    semantic_mask = torch.flatten(psemask, start_dim = 1).permute(1, 0)  # class x h x w -> (h x w) x class
    max_idx = semantic_mask.argmax(1) # (h x w)
    # print(max_idx.max())
    # output_masks = torch.zeros(semantic_mask.shape).scatter(1, max_idx.unsqueeze(1), 1.0) # one hot: (h x w) x class
    # output_masks = torch.reshape(output_masks.permute(1, 0), (nc, h, w)) # (h x w) x class -> class x h x w
    # cc = np.logical_and(masks[0].numpy, masks[1].numpy)
    # cv2.imshow('ddd',cc*255)
    # cv2.waitKey()

    semask = max_idx.view(h0, w0)
    panoptic = panoptic_merge_show(semask.numpy(), masks.numpy(), pred[0][:, 5].numpy(), pred[0][:, 4].numpy(), min_area=64*64)
    panoptic = crop_image(panoptic.shape[:2], panoptic, img_raw.shape)
    panoptic = cv2.resize(panoptic, (img_raw.shape[1], img_raw.shape[0]), interpolation=cv2.INTER_NEAREST_EXACT)
    # print(np.unique(panoptic[:,:,0]))
    # print(np.unique(panoptic[:,:,1]))
    # print(np.unique(panoptic[:,:,2]))
    # for image show
    color_image = np.zeros(panoptic.shape, dtype=np.uint8)
    for y in range(panoptic.shape[0]):
        for x in range(panoptic.shape[1]):
            semantic_id = panoptic[y, x, 2]
            if semantic_id != 0:
                color = colors((semantic_id // 1000) + (semantic_id % 1000))
            else:
                color = (0, 0, 0)  # black
            color_image[y, x] = color
    color_image = scale_image(color_image.shape[:2], color_image, img_raw.shape)

    # convert to save
    save_results = {'annotations': []}
    segments_info = []
    panoptic_images = {}
    unique_cls = np.unique(panoptic[:, :, 2])
    img_id = os.path.splitext(os.path.basename(img_path))[0]
    for _cls in unique_cls:
        if _cls > 0: # 0 is unlabeled
            annots = {}
            annots['category_id'] = int(_cls // 1000 if _cls > 183 else _cls)
            annots['id'] = int(_cls)
            segments_info.append(annots)
    save_results['annotations'].append({'img_id': img_id,
                                        'segments_info': segments_info})
    save_results_json = json.dumps(save_results, indent=4)
    file_name = img_id + ".txt"
    img_name = img_id + ".png"
    os.makedirs("../1_scripts/run/", exist_ok=True)
    file_path = os.path.join("../1_scripts/run/",file_name)
    with open(file_path, 'w') as f:
        f.write(save_results_json)
    panoptic[:, :, 1] = panoptic[:, :, 1] // 1000
    panoptic[:, :, 2] = panoptic[:, :, 2] % 256
    panoptic = panoptic.astype('uint8')
    panoptic_images[img_id] = panoptic
    cv2.imwrite(os.path.join("../1_scripts/run/",img_name), panoptic)

    unique_labels, inverse_indices = torch.unique(max_idx, return_inverse=True)  # shape: [N] 和 [H*W]
    inverse_indices = inverse_indices.view(h0, w0)
    N = unique_labels.size(0)
    output_masks = torch.zeros((N, h0, w0))
    output_masks = (inverse_indices.unsqueeze(0) == torch.arange(N).unsqueeze(1).unsqueeze(2)).byte()

    annotator_pan = Annotator(im_pan, line_width=3, example=str('semantic window'))
    annotator_pan.masks(output_masks,
                        colors=[colors(x, True) for x in unique_labels],
                        alpha=1)
    im_pan = annotator_pan.result()

    # process instance mask
    annotator = Annotator(img_raw, line_width=3, example=str('instance window')) 
    annotator.masks(masks, colors=[colors(x, True) for x in pred[0][:, 5]],)
    for _, (*xyxy, conf, cls) in enumerate(reversed(pred[0][:, :6])):
        c = int(cls)  # integer class
        label = f'{COCO_CLASSES[c]} {conf:.2f}'
        annotator.box_label(xyxy, label, color=colors(c, True))
    img_raw = annotator.result()

    cv2.imshow("semantic window", im_pan)
    cv2.imshow("instance window", img_raw)
    cv2.imshow('Panoptic Image', color_image)
    cv2.waitKey()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="../2_compile/fmodel/gelan-c-pan-640x640.pt", help='torchscript model path')
    parser.add_argument('--source', type=str, default='../0_yolov9/data/images/139.jpg', help='image path')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640], help='image size')
    opt = parser.parse_args()
    
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    
    pred_one_image(opt.source, opt.model)