import argparse
import os 
import cv2 
import torch 
import numpy as np 
import sys 
sys.path.append(R'../0_yolov10')
from ultralytics.nn.modules.block import DFL
from ultralytics.utils.tal import dist2bbox,make_anchors
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from visualize import vis,COCO_CLASSES

def pred_one_image(img_path,model_path,test_size):
    img_raw = cv2.imread(img_path)
    # 前处理
    letterbox = LetterBox(test_size, auto=False, stride=32)
    im = np.stack([letterbox(image=x) for x in [img_raw]])
    print('******im =',im.shape)

    im = im[..., ::-1].transpose((0, 3, 1, 2)) 
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im)
    im = im.float() 
    im /= 255
    # 加载traced模型
    model = torch.jit.load(model_path)
    output = model(im)
    for out in output:
        print(out.shape)
    print('*'*80)
    # 结果重组
    outputs_n1 = torch.cat((output[1], output[0]), 1)
    outputs_n2 = torch.cat((output[3], output[2]), 1)
    outputs_n3 = torch.cat((output[5], output[4]), 1)
    outputs = []
    outputs.append(outputs_n1)
    outputs.append(outputs_n2)
    outputs.append(outputs_n3)
    # for out in outputs:
    #     print(out.shape)
    # print('*'*80)
    # postprocess - dfl+sigmod
    shape = outputs[0].shape  # BCHW
    x_cat = torch.cat([xi.view(shape[0], 144, -1) for xi in outputs], 2)
    reg_max = 16
    nc = 80
    box, cls = x_cat.split((reg_max * 4, nc), 1)# box = [1,64,8400], cls = [1,80,8400]
    
    dfl_layer = DFL(reg_max)
    anchors, strides = (x.transpose(0, 1) for x in make_anchors(outputs, torch.from_numpy(np.array([8, 16, 32],dtype=np.float32)), 0.5))
    
    dbox = dist2bbox(dfl_layer(box), anchors.unsqueeze(0), xywh=True, dim=1) * strides
    
    y = torch.cat((dbox, cls.sigmoid()), 1) #[1,84,8400]
    
    # yolov10 postprocess - NMS free
    preds = y.transpose(-1, -2)
    
    conf_thres = 0.25
    max_det = 300
    bboxes, scores, labels = ops.v10postprocess(preds,max_det, preds.shape[-1]-4)# bbox - [1,max_det,4] scores - [1,max_det] labels - [1,300]

    bboxes = ops.xywh2xyxy(bboxes)
    
    preds = torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1) #[1,max_det,6] = [1,max_det, bbox+scores+label]
    mask = preds[..., 4] > conf_thres
    
    b, _, c = preds.shape
    preds = preds.view(-1, preds.shape[-1])[mask.view(-1)]# 取mask = True的结果，即score>conf的结果
    pred = preds.view(b, -1, c)#[1,res_num,6]
    _,res_num,_ = pred.shape
    pred = pred[0]
    
    # rescale coords to img_raw size
    pred[:, :4] = ops.scale_boxes(im.shape[2:], pred[:, :4], img_raw.shape)
    # show results
    result_image = vis(img_raw, boxes=pred[:,:4], scores=pred[:,4], cls_ids=pred[:,5], conf=conf_thres, class_names=COCO_CLASSES)
    cv2.imshow(" ", result_image)
    cv2.waitKey(0)
    print('Detect ',res_num,' objects!')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="../2_compile/fmodel/yolov10n_640x640.pt", help='torchscript model path')
    parser.add_argument('--source', type=str, default="../2_compile/qtset/coco/000000001000.jpg", help='image path')
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