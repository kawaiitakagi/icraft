import cv2
import time
import torch
import numpy as np
import sys
sys.path.append(R"../0_ultralytics")
sys.path.append(R"../0_yolov5_seg")
from utils.augmentations import letterbox
from utils.general import check_version,non_max_suppression,scale_boxes
from utils.segment.general import process_mask
from visualize import vis, COCO_CLASSES # 可视化函数借鉴的YoloX/utils/visualize.py
from utils.plots import Colors

test_size = (640, 640)
IMAGE_PATH = R"../2_compile/qtset/coco/000000000632.jpg"
TRACED_MODEL_PATH = "../2_compile/fmodel/YoloV5s_seg_640x640.pt"
stride = [8, 16, 32]
anchors = torch.tensor([[[10,13], [16,30], [33,23]], [[30,61], [62,45], [59,119]], [[116,90], [156,198], [373,326]]])


def make_grid( nx=20, ny=20, i=0):
    shape = 1, 3, ny, nx, 2  # grid shape
    y, x = torch.arange(ny), torch.arange(nx)
    if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
        yv, xv = torch.meshgrid(y, x, indexing='ij')
    else:
        yv, xv = torch.meshgrid(y, x)
    grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
    anchor_grid = anchors[i].view((1, 3, 1, 1, 2)).expand(shape)
    return grid, anchor_grid

# 读取图片
img_raw = cv2.imread(IMAGE_PATH)

# 前处理
img = letterbox(img_raw, test_size, stride=32, auto=False)[0]
img = img.transpose((2, 0, 1))[::-1] # HWC to CHW, BGR to RGB
img = np.ascontiguousarray(img)
im = torch.from_numpy(img).float().unsqueeze(0)
im /= 255

# 加载traced模型
model = torch.jit.load(TRACED_MODEL_PATH)
t0 = time.time()
outputs = model(im)
t1 = time.time()
# print(t1-t0)

# 后处理
z = []
grid = [torch.zeros(1)] * 3
anchor_grid = [torch.zeros(1)] * 3

for i in range(3):
    bs, _ , ny, nx = outputs[i].shape
    outputs[i] = outputs[i].view(1, 3, 117, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

    grid[i], anchor_grid[i] = make_grid(nx, ny, i)

    y = outputs[i]
    y[...,0:85] = y[...,0:85].sigmoid()
    y[..., 0:2] = (y[..., 0:2] * 2 + grid[i]) * stride[i]  # xy
    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
    z.append(y.view(bs, -1, 117))


pred = torch.cat(z, 1)
proto = outputs[3]

conf_thres = 0.25
iou_thres = 0.45

pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False, max_det=1000, nm=32)

det = pred[0]

masks = process_mask(proto[0], det[:, 6:], det[:, :4], [640,640], upsample=True)  # HWC

det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img_raw.shape).round()  # rescale boxes to im0 size

def scale_image(im1_shape, masks, im0_shape, ratio_pad=None):
    """
    img1_shape: model input shape, [h, w]
    img0_shape: origin pic shape, [h, w, 3]
    masks: [h, w, num]
    """
    # Rescale coordinates (xyxy) from im1_shape to im0_shape
    if ratio_pad is None:  # calculate from im0_shape
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
    else:
        pad = ratio_pad[1]
    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])

    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    masks = masks[top:bottom, left:right]
    # masks = masks.permute(2, 0, 1).contiguous()
    # masks = F.interpolate(masks[None], im0_shape[:2], mode='bilinear', align_corners=False)[0]
    # masks = masks.permute(1, 2, 0).contiguous()
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))

    if len(masks.shape) == 2:
        masks = masks[:, :, None]
    return masks

def masks_fuction(im,masks, colors, im_gpu, alpha=0.5, retina_masks=False):
    colors = torch.tensor(colors, device=im_gpu.device, dtype=torch.float32) / 255.0
    colors = colors[:, None, None]  # shape(n,1,1,3)

    masks = masks.unsqueeze(3)  # shape(n,h,w,1)

    masks_color = masks * (colors * alpha)  # shape(n,h,w,3)

    inv_alph_masks = (1 - masks * alpha).cumprod(0)  # shape(n,h,w,1)
    mcs = (masks_color * inv_alph_masks).sum(0) * 2  # mask color summand shape(n,h,w,3)
    im_gpu = im_gpu.flip(dims=[0])  # flip channel
    im_gpu = im_gpu.permute(1, 2, 0).contiguous()  # shape(h,w,3)

    im_gpu = im_gpu * inv_alph_masks[-1] + mcs
    im_mask = (im_gpu * 255).byte().cpu().numpy()
    im[:] = im_mask if retina_masks else scale_image(im_gpu.shape, im_mask, im.shape)
    return np.asarray(im)

colors = Colors()  # create instance for 'from utils.plots import colors'
img_raw = masks_fuction(img_raw, masks, colors=[colors(x, True) for x in det[:, 5]], im_gpu=im[0])
result_image = vis(img_raw, boxes=det[:,:4], scores=det[:,4], cls_ids=det[:,5], conf=conf_thres, class_names=COCO_CLASSES)
cv2.imshow(" ", result_image)
cv2.waitKey(0)



