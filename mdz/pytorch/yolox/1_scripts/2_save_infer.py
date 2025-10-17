import sys
sys.path.append(R"../0_yolox")
import cv2
import os
import torch
import argparse
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import postprocess, vis
from yolox.utils import meshgrid

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_pt', type=str, default=R"../2_compile/fmodel/yolox_s_640x640_traced.pt", help='torchscript model path')
    parser.add_argument('--source', type=str, default=R"../0_yolox/assets/dog.jpg", help='image path')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640], help='image size')
    opt = parser.parse_args()
    return opt



def decode_outputs(outputs, dtype):
    grids = []
    strides = []
    strides_1 = [8, 16, 32]
    hw = [(80, 80), (40, 40), (20, 20)] 
    for (hsize, wsize), stride in zip(hw, strides_1):
        yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
        grid = torch.stack((xv, yv), 2).view(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        strides.append(torch.full((*shape, 1), stride))

    grids = torch.cat(grids, dim=1).type(dtype)
    strides = torch.cat(strides, dim=1).type(dtype)

    outputs[..., :2] = (outputs[..., :2] + grids) * strides
    outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
    return outputs

def pred_one_image(image_path, model_path, test_size):
    # 读取图片
    img_raw = cv2.imread(image_path)

    # 前处理
    preproc = ValTransform()
    img, _ = preproc(img_raw, None, test_size)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.float()

    # 加载traced模型
    
    model = torch.jit.load(model_path)
    outputs_1 = model(img)

    # 后处理
    outputs = []
    for i in range(3):
        obj_output = outputs_1[0 + 3*i]
        reg_output = outputs_1[1 + 3*i]
        cls_output = outputs_1[2 + 3*i]
        output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)
        outputs.append(output)
    # [batch, n_anchors_all, 85]
    outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
    decode_outputs(outputs, dtype=torch.float32)

    # NMS
    conf_thres = 0.25
    iou_thres = 0.45
    outputs = postprocess(outputs, num_classes=80, conf_thre=conf_thres, nms_thre=iou_thres, class_agnostic=True)

    bboxes = outputs[0][:, 0:4]
    ratio = min(test_size[0] / img_raw.shape[0], test_size[1] / img_raw.shape[1])
    bboxes /= ratio
    scores = outputs[0][:, 4] * outputs[0][:, 5]
    cls = outputs[0][:,6]

    result_image = vis(img_raw, bboxes, scores, cls, conf=conf_thres, class_names=COCO_CLASSES)
    cv2.imshow(" ", result_image)
    cv2.waitKey(0)
    cv2.imwrite("./outputs/save_infer.jpg", result_image)

if __name__ == "__main__":
    opt = parse()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    test_size = tuple(opt.imgsz)
    if os.path.isfile(opt.source):
        pred_one_image(opt.source, opt.model_pt, test_size)
    elif os.path.isdir(opt.source):
        image_list = os.listdir(opt.source)
        for image_file in image_list:
            image_path = opt.source + "//" + image_file 
            pred_one_image(image_path, opt.model_pt, test_size)

    