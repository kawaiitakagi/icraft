from distutils.command import sdist
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from pathlib import Path


# when you do not want to import torch 

class DFL:
    # DFL module
    def __init__(self, c1=17):
        self.c1 = c1
        self.conv_weights = np.arange(c1, dtype=np.float32).reshape(1, c1, 1, 1)
        
    def softmax(self, x, axis):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def conv2d(self, x, weights):
        # Assuming a 1x1 convolution for simplicity
        return np.sum(x * weights, axis=1, keepdims=True)

    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        x = x.reshape(b, 4, self.c1, a).transpose(0, 2, 1, 3)  # reshape and transpose
        x = self.softmax(x, axis=1)  # apply softmax along the specified axis
        x = self.conv2d(x, self.conv_weights)  # apply 1x1 convolution
        return x.reshape(b, 4, a)  # reshape back to the desired output shape
    
def make_anchors(feats, strides, grid_cell_offset=0.5):
    # Generate anchors from features using NumPy
    anchor_points = []
    stride_tensor = []
    assert feats is not None
    
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = np.arange(w) + grid_cell_offset  # shift x
        sy = np.arange(h) + grid_cell_offset  # shift y
        sy, sx = np.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(np.stack((sx, sy), -1).reshape(-1, 2))
        stride_tensor.append(np.full((h * w, 1), stride))
    
    return np.concatenate(anchor_points), np.concatenate(stride_tensor)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    # Transform distance(ltrb) to box(xywh or xyxy) using NumPy
    lt, rb = np.split(distance, 2, axis=dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return np.concatenate((c_xy, wh), axis=dim)  # xywh bbox
    return np.concatenate((x1y1, x2y2), axis=dim)  # xyxy bbox

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def nms(boxes, scores, iou_threshold):
    """
    Implements Non-Maximum Suppression (NMS) to filter the predictions.
    
    Args:
        boxes (numpy.ndarray): Array of shape (n, 4) where n is the number of predicted boxes.
                               Each row represents [x1, y1, x2, y2].
        scores (numpy.ndarray): Array of shape (n,) where n is the number of predicted boxes.
                                Each element represents the confidence score of the corresponding box.
        iou_threshold (float): IOU threshold for NMS.
        
    Returns:
        List[int]: List of indices of the boxes to keep.
    """
    
    # If no boxes, return an empty list
    if len(boxes) == 0:
        return []
    
    # Get the coordinates of the boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # Compute the area of the boxes
    areas = (x2 - x1) * (y2 - y1)
    
    # Sort the boxes by scores in descending order
    order = scores.argsort()[::-1]
    
    keep = []  # List to keep the indices of boxes to keep
    
    # Iterate through all boxes
    for i in range(len(order)):
        if order.size == 0:
            break
        
        idx = order[0]  # Index of the current box with the highest score
        keep.append(idx)
        
        # Get the coordinates of the intersection boxes
        xx1 = np.maximum(x1[idx], x1[order[1:]])
        yy1 = np.maximum(y1[idx], y1[order[1:]])
        xx2 = np.minimum(x2[idx], x2[order[1:]])
        yy2 = np.minimum(y2[idx], y2[order[1:]])
        
        # Compute the width and height of the intersection boxes
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        # Compute the intersection over union (IoU)
        inter = w * h
        iou = inter / (areas[idx] + areas[order[1:]] - inter)
        
        # Keep boxes with IoU less than the threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep

def compute_iou(boxes1, boxes2):
    """Compute IOU between two sets of boxes."""
    # Compute intersection area
    x1 = np.maximum(boxes1[:, 0], boxes2[:, 0])
    y1 = np.maximum(boxes1[:, 1], boxes2[:, 1])
    x2 = np.minimum(boxes1[:, 2], boxes2[:, 2])
    y2 = np.minimum(boxes1[:, 3], boxes2[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

    # Compute union area
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - intersection

    # Compute IOU
    iou = intersection / np.maximum(union, 1e-6)
    return iou


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    if isinstance(prediction, (list, tuple)):  # YOLO model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[1] - nm - 4  # number of classes (84+32-32-4)
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].max(1) > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    output = [np.zeros((0, 6 + nm))] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x.T[xc[xi]]  # confidence   

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = np.zeros((len(lb), nc + nm + 5))
            v[:, :4] = lb[:, 1:5]  # box
            v[np.arange(len(lb)), lb[:, 0].astype(int) + 4] = 1.0  # cls
            x = np.concatenate((x, v), axis=0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box = x[:, :4]
        cls = x[:, 4:4 + nc]
        mask = x[:, 4 + nc:]

        box = xywh2xyxy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        if multi_label:
            i, j = np.nonzero(cls > conf_thres)
            x = np.concatenate((box[i], x[i, 4 + j, None], j[:, None].astype(float), mask[i]), axis=1)
        else:  # best class only
            conf = np.amax(cls, axis=1, keepdims=True)
            j = np.argmax(cls, axis=1, keepdims=True).astype(float)
            x = np.concatenate((box, conf, j, mask), axis=1) # 4+1+1+32 = 38
            x = x[conf.flatten() > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes)).any(axis=1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort()[::-1][:max_nms]]  # sort by confidence
        else:
            x = x[x[:, 4].argsort()[::-1]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * max_wh # classes
        boxes, scores = x[:, :4] + c, x[:, 4] # boxes(offset by class), scores
        i = nms(boxes, scores, iou_thres)

        if len(i) > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]

    return output

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

def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    # np.array (faster grouped)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    boxes[:, :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def crop_mask(masks, boxes):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.

    Args:
        - masks: a size [h, w, n] array of masks
        - boxes: a size [n, 4] array of bbox coords in relative point form

    Returns:
        - cropped_masks: a size [h, w, n] array of cropped masks
    """

    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, None, :], 4, axis=2)  # x1 shape(1,1,n)
    r = np.arange(w)[None, None, :]  # rows shape(1,w,1)
    c = np.arange(h)[None, :, None]  # cols shape(h,1,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """
    Crop before upsample.
    protos: [mask_dim, mask_h, mask_w]
    masks_in: [n, mask_dim], n is number of masks after NMS
    bboxes: [n, 4], n is number of masks after NMS
    shape: input_image_size, (h, w)
    upsample: Whether to upsample the masks

    Returns:
    masks: [h, w, n]
    """

    c, mh, mw = protos.shape  # CHW

    ih, iw = shape 
    # mask_in check is ok!
    masks = sigmoid(np.dot(masks_in, protos.reshape(c, -1))).reshape(-1, mh, mw)
    # masks check is ok!

    downsampled_bboxes = bboxes.copy()
    downsampled_bboxes[:, 0] *= mw / iw
    downsampled_bboxes[:, 2] *= mw / iw
    downsampled_bboxes[:, 3] *= mh / ih
    downsampled_bboxes[:, 1] *= mh / ih
    
    masks = crop_mask(masks, downsampled_bboxes)  # nHW
    
    if upsample:
        masks = cv2.resize(masks.transpose(1, 2, 0), (iw, ih), interpolation=cv2.INTER_LINEAR)  # HWn
    return (masks > 0.5).astype(np.uint8)  # Convert to binary masks

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'

def box_label(im, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(im, p1, p2, color, thickness=2 , lineType=cv2.LINE_AA)
    if label:
        tf = 2  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=3 / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im,
                    label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    3 / 3,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)
    return np.asarray(im)

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
    # masks shape: [h, w, n]
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))

    if len(masks.shape) == 2:
        masks = masks[:, :, None]
    return masks

def crop_image(im1_shape, masks, im0_shape, ratio_pad=None):
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
    return masks

def draw_masks(im, masks, colors, alpha=0.5):
    """Plot masks at once.
    Args:
        masks (array): predicted masks , shape: [h, w, n]
        colors (List[List[Int]]): colors for predicted masks, [[r, g, b] * n]
        alpha (float): mask transparency: 0.0 fully transparent, 1.0 opaque
    """
    if len(masks) == 0:
        return
    masks = scale_image(masks.shape[:2], masks, im.shape) # scale masks to im's shape
    masks = np.asarray(masks, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32) # shape(n,3)
    s = masks.sum(2, keepdims=True).clip(0, 1) # add all masks together
    masks = (masks @ colors).clip(0, 255) # (h,w,n) @ (n,3) = (h,w,3)
    im[:] = masks * alpha + im * (1 - s * alpha)
    return np.asarray(im)


# coco id
all_instances_ids = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 27, 28,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
    61, 62, 63, 64, 65, 67, 70,
    72, 73, 74, 75, 76, 77, 78, 79, 80,
    81, 82, 84, 85, 86, 87, 88, 89, 90,
]

all_stuff_ids = [
    92, 93, 94, 95, 96, 97, 98, 99, 100,
    101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
    111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
    121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
    131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
    141, 142, 143, 144, 145, 146, 147, 148, 149, 150,
    151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
    161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
    171, 172, 173, 174, 175, 176, 177, 178, 179, 180,
    181, 182,
    # other
    183,
    # unlabeled
    0,
]   

# panoptic id: https://github.com/cocodataset/panopticapi/blob/master/panoptic_coco_categories.json
panoptic_stuff_ids = [
    92, 93, 95, 100,
    107, 109,
    112, 118, 119,
    122, 125, 128, 130,
    133, 138,
    141, 144, 145, 147, 148, 149,
    151, 154, 155, 156, 159,
    161, 166, 168,
    171, 175, 176, 177, 178, 180,
    181, 184, 185, 186, 187, 188, 189, 190,
    191, 192, 193, 194, 195, 196, 197, 198, 199, 200,
    # unlabeled
    0,
]


def getCocoIds(name = 'semantic'):
    if 'instances' == name:
        return all_instances_ids
    elif 'stuff' == name:
        return all_stuff_ids
    elif 'panoptic' == name:
        return all_instances_ids + panoptic_stuff_ids
    else: # semantic
        return all_instances_ids + all_stuff_ids

def getMappingId(index, name = 'semantic'):
    ids = getCocoIds(name = name)
    return ids[index]

def getMappingIndex(id, name = 'semantic'):
    ids = getCocoIds(name = name)
    return ids.index(id)

def panoptic_merge_show(semask, masks, labels, conf, min_area):
    panoptic = np.zeros(semask.shape + (3,), dtype=np.int32)
    stuff = np.zeros_like(semask)
    unique_labels = np.unique(semask)
    for _cls in unique_labels:
        if _cls < 80:
            stuff[semask == _cls] = 255
        else:
            stuff[semask == _cls] = getMappingId(_cls, 'semantic')
    panoptic[:, :, 2] = stuff
    panoptic[:, :, 0] = stuff
    inst_id = 0

    # merge inst
    area = np.sum(masks, axis=(1,2))
    # print(area.shape)
    sorted_indices = np.argsort(area)[::-1]
    # # print("**", sorted_indices)
    masks = masks[sorted_indices]
    labels = labels[sorted_indices]
    conf = conf[sorted_indices]
    # used = None
    # print(labels)
    for i in range(len(masks)):
  
        valid_area = (masks[i] == 1)
        panoptic[:, :, 1][valid_area] = getMappingId(int(labels[i]), 'instances') * 1000 + inst_id
        panoptic[:, :, 2][valid_area] = getMappingId(int(labels[i]), 'instances') * 1000 + inst_id
        panoptic[:, :, 0][valid_area] = getMappingId(int(labels[i]), 'instances')
        inst_id += 1
    
    # for _cls in np.unique(getMappingId(int(labels[i]), 'instances')):
    #     inst_id = 0
    #     for i in range(len(masks)):
    #         if labels[i] == _cls:
    #             valid_area = (masks[i] == 1)
    #             panoptic[:, :, 1][valid_area] = getMappingId(int(labels[i]), 'instances') * 1000 + inst_id
    #             panoptic[:, :, 2][valid_area] = getMappingId(int(labels[i]), 'instances') * 1000 + inst_id
    #             inst_id += 1
        
    # merge stuff
    stuff_map = panoptic[:, :, 1] == 0
    stuff_cls = np.unique(panoptic[:, :, 2][stuff_map])
    for _cls in stuff_cls:
        stuff_seg = (panoptic[:, :, 2] == _cls).astype(np.uint8)
        num, componets = cv2.connectedComponents(stuff_seg)
        for i in range(num):
            if i > 0:
                com_map = componets == i
                if np.count_nonzero(com_map) <= min_area:
                    panoptic[:, :, 2][com_map] = 255
                    panoptic[:, :, 0][com_map] = 255

    # Convert 255 to Unlabeled
    panoptic[panoptic == 255] = 0
    # panoptic[:, :, 1] = panoptic[:, :, 1] // 256
    # panoptic[:, :, 2] = panoptic[:, :, 2] % 256
    # panoptic = panoptic.astype('uint8')
    return panoptic


def panoptic_merge_coco(semask, masks, labels, min_area):
    panoptic = np.zeros(semask.shape + (3,), dtype=np.int32)
    stuff = np.zeros_like(semask)
    unique_labels = np.unique(semask)
    for _cls in unique_labels:
        if _cls < 92:
            stuff[semask == _cls] = 255
        else:
            stuff[semask == _cls] = _cls
    panoptic[:, :, 2] = stuff
    panoptic[:, :, 0] = stuff
    
    # merge inst
    inst_id = 0
    for i in range(len(masks)):
        valid_area = (masks[i] == 1)
        panoptic[:, :, 1][valid_area] = labels[i] * 1000 + inst_id
        # print(labels[i])
        # print('debug.......',np.unique(panoptic[:, :, 1]))
        panoptic[:, :, 2][valid_area] = labels[i] * 1000 + inst_id
        panoptic[:, :, 0][valid_area] = labels[i]
        inst_id += 1
    # print('debug.......',np.unique(panoptic[:, :, 1]))

    # for _cls in np.unique(labels):
    #     inst_id = 0
    #     imasks = masks[labels == _cls]
    #     for i, inst in enumerate(imasks):
    #         valid_area = (inst == 1)
    #         panoptic[:, :, 1][valid_area] = _cls * 1000 + inst_id
    #         panoptic[:, :, 2][valid_area] = _cls * 1000 + inst_id
    #         inst_id += 1


    # merge stuff
    stuff_map = panoptic[:, :, 1] == 0
    stuff_cls = np.unique(panoptic[:, :, 2][stuff_map])
    for _cls in stuff_cls:
        stuff_seg = (panoptic[:, :, 2] == _cls).astype(np.uint8)
        num, componets = cv2.connectedComponents(stuff_seg)
        for i in range(num):
            if i > 0:
                com_map = componets == i
                if np.count_nonzero(com_map) <= min_area:
                    panoptic[:, :, 2][com_map] = 255

    # Convert 255 to Unlabeled
    panoptic[panoptic == 255] = 0
    # panoptic[:, :, 1] = panoptic[:, :, 1] // 256
    # panoptic[:, :, 2] = panoptic[:, :, 2] % 256
    # panoptic = panoptic.astype('uint8')
    return panoptic

