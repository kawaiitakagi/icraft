IMAGE_PATH = "../2_compile/qtset/coco/000000000632.jpg"
TRACED_MODEL_PATH = "../2_compile/fmodel/yolov8n_640x640.pt"
import sys
sys.path.append(R"../0_yolov8")
import argparse
import os
import cv2
import torch
import numpy as np
from DFL import DFL
from ultralytics.utils.tal import dist2bbox, make_anchors
from ultralytics.data.augment import LetterBox
from ultralytics.utils import  ops
from visualize import vis, COCO_CLASSES
def pred_one_image(img_path, model_path, test_size):

    img_raw = cv2.imread(img_path)

    # 前处理
    # 前处理
    img_resize = cv2.resize(img_raw, (640,640))
    # img = LetterBox(new_shape=test_size, auto=True, stride=32)(image=img_raw)
    # img = letterbox(img_raw, new_shape=test_size, stride=32, auto=True)[0]  # auto参数设为True表示带pad的resize
    img = img_resize.transpose((2, 0, 1))[::-1] # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    im = torch.from_numpy(img).float().unsqueeze(0)
    im /= 255

    # 加载traced模型
    model = torch.jit.load(model_path)
    output = model(im)

    # 后处理

    outputs_n1 = torch.cat((output[1], output[0]), 1)
    outputs_n2 = torch.cat((output[3], output[2]), 1)
    outputs_n3 = torch.cat((output[5], output[4]), 1)
    outputs = []
    outputs.append(outputs_n1)
    outputs.append(outputs_n2)
    outputs.append(outputs_n3)
    # dfl+sigmod
    reg_max = 16
    nc = 80
    dfl_layer = DFL(reg_max)
    anchors, strides = (x.transpose(0, 1) for x in make_anchors(outputs, torch.from_numpy(np.array([8, 16, 32],dtype=np.float32)), 0.5))
    box, cls = torch.cat([xi.view(output[1].shape[0], reg_max*4+80, -1) for xi in outputs], 2).split((reg_max*4, nc), 1)
    dbox = dist2bbox(dfl_layer(box), anchors.unsqueeze(0), xywh=True, dim=1) * strides
    y = torch.cat((dbox, cls.sigmoid()), 1)
    # postprocess
    conf_thres = 0.25
    iou_thres = 0.7
    preds = ops.non_max_suppression(y,
                                    conf_thres,
                                    iou_thres,
                                    agnostic=False,
                                    max_det=300,
                                    classes=None)
    # visual
    ratio = min(640 / img_raw.shape[0], 640 / img_raw.shape[1])
    preds[0][:, :4] /= ratio
    result_image = vis(img_resize, boxes=preds[0][:,:4], scores=preds[0][:,4], cls_ids=preds[0][:,5], conf=conf_thres, class_names=COCO_CLASSES)
    cv2.imshow(" ", result_image)
    cv2.waitKey(0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=TRACED_MODEL_PATH, help='torchscript model path')
    parser.add_argument('--source', type=str, default=IMAGE_PATH, help='image path')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640], help='image size')
    opt = parser.parse_args()
    
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    test_size = tuple(opt.imgsz)

    if os.path.isfile(opt.source):
        pred_one_image(opt.source, opt.model, test_size)
    elif os.path.isdir(opt.source):
        image_list = os.listdir(opt.source)
        for image_file in image_list:
            image_path = opt.source + "//" + image_file 
            pred_one_image(image_path, opt.model, test_size)
