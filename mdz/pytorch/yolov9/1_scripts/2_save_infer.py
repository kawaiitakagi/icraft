import sys
import argparse
import cv2
import torch
import numpy as np
sys.path.append(R"../0_yolov9")
from utils.tal.anchor_generator import make_anchors,dist2bbox
from models.common import DFL
from utils.general import non_max_suppression,scale_boxes
from utils.augmentations import letterbox
from visualize import vis, COCO_CLASSES

def pred_one_image(img_path, model_path):

    
    # 前处理
    img_raw = cv2.imread(img_path)
    img_resize = letterbox(img_raw, 640, stride=32, auto=False)[0]
    im = img_resize.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im).float().unsqueeze(0)
    im /= 255
    if len(im.shape) == 3:
        im = im[None] # expand for batch dim
    
    # 加载traced模型
    model = torch.jit.load(model_path)
    output = model(im)
    
    # 结果重组
    outputs_n1 = torch.cat((output[1], output[0]), 1) #[1,144,80,80]
    outputs_n2 = torch.cat((output[3], output[2]), 1) #[1,144,40,40]
    outputs_n3 = torch.cat((output[5], output[4]), 1) #[1,144,20,20]
    outputs = []
    outputs.append(outputs_n1)
    outputs.append(outputs_n2)
    outputs.append(outputs_n3)
    for out in outputs:
        print(out.shape)
    print('*'*80)
    
    # postprocess - dfl+sigmod
    reg_max = 16
    nc = 80
    dfl_layer = DFL(reg_max)
    anchors, strides = (x.transpose(0, 1) for x in make_anchors(outputs, torch.from_numpy(np.array([8, 16, 32],dtype=np.float32)), 0.5))
    box, cls = torch.cat([xi.view(output[1].shape[0], reg_max*4+80, -1) for xi in outputs], 2).split((reg_max*4, nc), 1)
    
    dbox = dist2bbox(dfl_layer(box), anchors.unsqueeze(0), xywh=True, dim=1) * strides
    y = torch.cat((dbox, cls.sigmoid()), 1)
    
    # postprocess - NMS
    conf_thres = 0.25
    iou_thres = 0.45
    pred = non_max_suppression(y,
                                conf_thres,
                                iou_thres,
                                agnostic=False,
                                max_det=300,
                                classes=None)

    #检测结果进行scale box,在原图显示 
    pred[0][:, :4] = scale_boxes(im.shape[2:], pred[0][:, :4], img_raw.shape).round()
    # show results
    result_image = vis(img_raw, boxes=pred[0][:,:4], scores=pred[0][:,4], cls_ids=pred[0][:,5], conf=conf_thres, class_names=COCO_CLASSES)
    cv2.imshow(" ", result_image)
    cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="../2_compile/fmodel/yolov9t_640x640.pt", help='torchscript model path')
    parser.add_argument('--source', type=str, default='../0_yolov9/data/images/horses.jpg', help='image path')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640], help='image size')
    opt = parser.parse_args()
    
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    
    pred_one_image(opt.source, opt.model)