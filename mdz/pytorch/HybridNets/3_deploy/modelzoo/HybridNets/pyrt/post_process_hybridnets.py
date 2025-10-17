import cv2
import numpy as np
import torch
import webcolors
from torchvision.ops.boxes import batched_nms


STANDARD_COLORS = [
    'LawnGreen', 'Chartreuse', 'Aqua', 'Beige', 'Azure', 'BlanchedAlmond', 'Bisque',
    'Aquamarine', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'AliceBlue', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

def read_list_from_txt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

def from_colorname_to_bgr(color):
    rgb_color = webcolors.name_to_rgb(color)
    result = (rgb_color.blue, rgb_color.green, rgb_color.red)
    return result

def standard_to_bgr(list_color_name):
    standard = []
    for i in range(len(list_color_name) - 36):  # -36 used to match the len(obj_list)
        standard.append(from_colorname_to_bgr(list_color_name[i]))
    return standard

def letterbox(combination, new_shape=(384, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    """缩放并在图片顶部、底部添加灰边，具体参考：https://zhuanlan.zhihu.com/p/172121380"""
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    img, seg = combination
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        if seg:
            for seg_class in seg:
                seg[seg_class] = cv2.resize(seg[seg_class], new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    if seg:
        for seg_class in seg:
            seg[seg_class] = cv2.copyMakeBorder(seg[seg_class], top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)  # add border

    combination = (img, seg)
    return combination, ratio, (dw, dh)

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if len(coords) == 0:
        return []
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def plot_one_box(img, coord, label=None, score=None, color=None, line_thickness=None):
    tl = line_thickness or int(round(0.001 * max(img.shape[0:2])))  # line thickness
    color = color
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 2, 1)  # font thickness
        s_size = cv2.getTextSize(str('{:.0%}'.format(score)), 0, fontScale=float(tl) / 3, thickness=tf)[0]
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0] + s_size[0] + 15, c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, '{}: {:.0%}'.format(label, score), (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0],
                    thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)

def BBoxTransform_forward(anchors, regression):
    y_centers_a = (anchors[..., 0] + anchors[..., 2]) / 2
    x_centers_a = (anchors[..., 1] + anchors[..., 3]) / 2
    ha = anchors[..., 2] - anchors[..., 0]
    wa = anchors[..., 3] - anchors[..., 1]

    w = regression[..., 3].exp() * wa
    h = regression[..., 2].exp() * ha

    y_centers = regression[..., 0] * ha + y_centers_a
    x_centers = regression[..., 1] * wa + x_centers_a

    ymin = y_centers - h / 2.
    xmin = x_centers - w / 2.
    ymax = y_centers + h / 2.
    xmax = x_centers + w / 2.

    return torch.stack([xmin, ymin, xmax, ymax], dim=2)
      
def seg_postprocess(img_ori,seg,seg_list, color_list_seg,pad,MULTICLASS):
    if MULTICLASS:
        _, seg_mask = torch.max(seg, 1)
    # (B, W, H) -> (W, H)
    # print("seg postprocess size: ",seg.size(0))
    seg_mask_ = seg_mask.squeeze().cpu().numpy()#(384,640)
    pad_h = int(pad[1])
    pad_w = int(pad[0])
    seg_mask_ = seg_mask_[pad_h:seg_mask_.shape[0]-pad_h, pad_w:seg_mask_.shape[1]-pad_w]
    seg_mask_ = cv2.resize(seg_mask_, dsize=img_ori.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)#[720,1280]
    color_seg = np.zeros((seg_mask_.shape[0], seg_mask_.shape[1], 3), dtype=np.uint8)
    for index, seg_class in enumerate(seg_list):
        color_seg[seg_mask_ == index+1] = color_list_seg[seg_class]
    color_seg = color_seg[..., ::-1]  # RGB -> BGR
    # cv2.imwrite('seg_only_{}.jpg'.format(i), color_seg)

    color_mask = np.mean(color_seg, 2)  # (H, W, C) -> (H, W), check if any pixel is not background
    seg_img = img_ori.copy()
    seg_img[color_mask != 0] = seg_img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    seg_img = seg_img.astype(np.uint8)
    return seg_img

def det_postprocess(img_ori,regression,classification,shapes,obj_list,color_list,anchors,threshold,iou_threshold):
    transformed_anchors = BBoxTransform_forward(anchors, regression)
    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > threshold)[:, :, 0] #[1,46035]
    out = {'rois': np.array(()),
            'class_ids': np.array(()),
            'scores': np.array(())}
    classification_per = classification[0,scores_over_thresh[0, :], ...].permute(1, 0)
    transformed_anchors_per = transformed_anchors[0, scores_over_thresh[0, :], ...]
    scores_per = scores[0, scores_over_thresh[0, :], ...]
    scores_, classes_ = classification_per.max(dim=0)
    anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)
    if anchors_nms_idx.shape[0] != 0:
        classes_ = classes_[anchors_nms_idx]
        scores_ = scores_[anchors_nms_idx]
        boxes_ = transformed_anchors_per[anchors_nms_idx, :]
        out["rois"] = boxes_.cpu().numpy()
        out["class_ids"] = classes_.cpu().numpy()
        out["scores"] = scores_.cpu().numpy()

    out['rois'] = scale_coords(img_ori[:2], out['rois'], img_ori.shape[:2], shapes[1])
    
    for j in range(len(out['rois'])):
        x1, y1, x2, y2 = out['rois'][j].astype(int)
        obj = obj_list[out['class_ids'][j]]
        score = float(out['scores'][j])
        plot_one_box(img_ori, [x1, y1, x2, y2], label=obj, score=score, color=color_list[int(obj_list.index(obj))])

    return img_ori

def postprocess_hybridnets(img_ori, regression, classification, seg, seg_list, obj_list, color_list, color_list_seg, anchors, pad,shapes, MULTICLASS, threshold, iou_threshold, show, save, res_img_path):
    res_img = img_ori.copy()
    regression, classification, seg = torch.tensor(regression), torch.tensor(classification), torch.tensor(seg)
    # 分割后处理：车道线检测分割、可行使区域分割
    res_img = seg_postprocess(res_img,seg,seg_list, color_list_seg,pad,MULTICLASS)

    # 目标检测后处理
    res_img = det_postprocess(res_img,regression,classification,shapes,obj_list,color_list,anchors,threshold,iou_threshold)
    
    if show:
        cv2.imshow('result img', res_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if save:
        cv2.imwrite(res_img_path, res_img)
   
    


