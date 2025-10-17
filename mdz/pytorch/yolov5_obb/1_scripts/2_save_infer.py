# 该脚本用来加载traced模型并执行一张图像的前向推理
import sys
sys.path.append(R"../0_yolov5_obb")
import cv2
import time
import torch
import numpy as np
from utils.augmentations import letterbox
from utils.general import check_version, non_max_suppression_obb,scale_polys
from utils.rboxs_utils import rbox2poly

# 参数设置
test_size = (1024,1024)
model_path = R"../2_compile/fmodel/YoloV5m_obb_1024x1024.pt"
img_path = R'..\0_yolov5_obb\dataset\dataset_demo\images\P0032.png'


names = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 
        'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank',  
        'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 
        'container-crane']  # class names

def make_grid(nx=20, ny=20, i=0):
    d = anchors[i].device
    if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
        yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)], indexing='ij')
    else:
        yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)])
    grid = torch.stack((xv, yv), 2).expand((1, 3, ny, nx, 2)).float()
    anchor_grid = (anchors[i].clone() * stride[i]) \
        .view((1, 3, 1, 1, 2)).expand((1, 3, ny, nx, 2)).float()
    return grid, anchor_grid

# 读取图片
img_raw = cv2.imread(img_path)

# 前处理
img_raw = letterbox(img_raw, new_shape=test_size, stride=32, auto=False)[0]
img = letterbox(img_raw, new_shape=test_size, stride=32, auto=False)[0]
img = img.transpose((2, 0, 1))[::-1] # HWC to CHW, BGR to RGB
img = np.ascontiguousarray(img)
im = torch.from_numpy(img).float().unsqueeze(0)
im /= 255


# 加载traced模型
model = torch.jit.load(model_path)
t0 = time.time()
outputs = model(im)
t1 = time.time()
print(t1-t0)


stride = [8, 16, 32]
anchors = model.model._modules._python_modules['24'].anchors

# 后处理
z = []
grid = [torch.zeros(1)] * 3
anchor_grid = [torch.zeros(1)] * 3

for i in range(3):
    bs, _ , ny, nx = outputs[i].shape
    outputs[i] = outputs[i].view(bs, 3, 201, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

    grid[i], anchor_grid[i] = make_grid(nx, ny, i)
    y = outputs[i].sigmoid()
    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + grid[i]) * stride[i]  # xy
    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
    z.append(y.view(bs, -1, 201))

pred = torch.cat(z, 1)  #torch.Size([1, 45927, 201])


# NMS
conf_thres = 0.25
iou_thres = 0.45
pred = non_max_suppression_obb(pred, conf_thres, iou_thres, classes=None, agnostic=False, multi_label=True, max_det=1000)

# 结果显示
det = pred[0]
pred_poly = rbox2poly(det[:, :5])
pred_poly = scale_polys(im.shape[2:], pred_poly, [1024,1024])
det = torch.cat((pred_poly, det[:, -2:]), dim=1) # (n, [poly conf cls])

color=(200, 0, 100)
txt_color = (255,255,255)

for *poly, conf, cls in reversed(det):
    polygon_list = np.array([(poly[0], poly[1]), (poly[2], poly[3]), \
                    (poly[4], poly[5]), (poly[6], poly[7])], np.int32)
    out = cv2.drawContours(image=img_raw, contours=[polygon_list], contourIdx=-1, color=color, thickness=2)
    c = int(cls)
    label =  f'{names[c]} {conf:.2f}'
    tf = max(2 - 1, 1)  # font thicknes
    xmax, xmin, ymax, ymin = max(poly[0::2]), min(poly[0::2]), max(poly[1::2]), min(poly[1::2])
    x_label, y_label = int((xmax + xmin)/2), int((ymax + ymin)/2)
    w, h = cv2.getTextSize(label, 0, fontScale=2 / 3, thickness=tf)[0]  # text width, height
    cv2.rectangle(
                    out,
                    (x_label, y_label),
                    (x_label + w + 1, y_label + int(1.5*h)),
                    color, -1, cv2.LINE_AA
                )
    cv2.putText(out, label, (x_label, y_label + h), 0, 2 / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)

cv2.imshow('YoloV5m_obb_traced_Result',out)
cv2.waitKey(0)