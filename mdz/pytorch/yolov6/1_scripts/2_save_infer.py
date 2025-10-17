import cv2
import time
import torch
import numpy as np
import sys 
sys.path.append(R"../0_yolov6")
from yolov6.data.data_augment import letterbox
from yolov6.assigners.anchor_generator import generate_anchors
from yolov6.utils.general import dist2bbox
from yolov6.utils.nms import non_max_suppression
from yolov6.core.inferer import Inferer


#----------------------visualize-------------------
# 可视化函数借鉴的YoloX/utils/visualize.py
def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)

COCO_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

test_size = (640, 640)
stride = [8, 16, 32]
model_path = R"../2_compile/fmodel/yolov6s_v2_reopt_640x640.pt"
img_path = R"../0_yolov6/data/images/image1.jpg"

# 读取图片
img_raw = cv2.imread(img_path)

# 前处理
img = letterbox(img_raw, new_shape=test_size, stride=32, auto=False)[0]
img = img.transpose((2, 0, 1))[::-1] # HWC to CHW, BGR to RGB
img = np.ascontiguousarray(img)
im = torch.from_numpy(img).float().unsqueeze(0)
im /= 255

# 加载traced模型
model = torch.jit.load(model_path)
t0 = time.time()
out = model(im)
t1 = time.time()
print(t1-t0)

# 后处理
cls_score_list,reg_dist_list = [],[]

bbox = []

for id in range(0,6,2):
    ftmp = out[id]
    ftmp = ftmp.sigmoid()
    b, _, h, w = ftmp.shape
    ftmp = ftmp.reshape([1, 80, h*w]).permute(0, 2, 1)
    cls_score_list.append(ftmp)

    shift_x = torch.arange(end=w) + 0.5
    shift_y = torch.arange(end=h) + 0.5
    shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
    anchor_point = torch.stack(
                    [shift_x, shift_y], axis=-1).to(torch.float).reshape([-1, 2]).unsqueeze(0)
    
    stride_tensor = torch.full(
                    (h * w, 1), stride[int(id/2)], dtype=torch.float)
    
    reg = out[id+1]
    b, _, h, w = reg.shape
    reg = reg.reshape([1, 4, h*w]).permute(0, 2, 1)
    reg_dist_list.append(reg)

    lt,rb = reg[...,:2],reg[...,2:4]

    c_xy = anchor_point + ( rb - lt)*0.5
    wh = reg[...,2:4] + reg[...,:2]
    out1 = torch.cat([c_xy, wh], -1)

    out1 = out1 * stride_tensor
    bbox.append(out1)

bbox = torch.cat(bbox, axis=1)

cls_score_list = torch.cat(cls_score_list, axis=1)

pred_results = torch.cat(
        [
            bbox,
            torch.ones((1, bbox.shape[1], 1), device=bbox.device, dtype=bbox.dtype),
            cls_score_list
        ],
        axis=-1)

# NMS
det = non_max_suppression(pred_results, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=1000)[0]
# 结果显示
det[:, :4] = Inferer.rescale(im.shape[2:], det[:, :4], img_raw.shape).round()
result_image = vis(img_raw, boxes=det[:,:4], scores=det[:,4], cls_ids=det[:,5], conf=0.4, class_names=COCO_CLASSES)
cv2.imshow(" ", result_image)
cv2.waitKey(0)
