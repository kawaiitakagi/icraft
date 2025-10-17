import cv2
import torch
import numpy as np
from general import *

def read_list_from_txt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

def get_Stride(netinfo):
    strides = []
    ih = netinfo.i_cubic[0].h
    for i in netinfo.o_cubic:
        strides.append(ih // i.h)
    return strides

def resize_unscale(img, new_shape=(640, 640), color=114):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    canvas = np.zeros((new_shape[0], new_shape[1], 3))
    canvas.fill(color)
    # Scale ratio (new / old) new_shape(h,w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # w,h
    new_unpad_w = new_unpad[0]
    new_unpad_h = new_unpad[1]
    pad_w, pad_h = new_shape[1] - new_unpad_w, new_shape[0] - new_unpad_h  # wh padding

    dw = pad_w // 2  # divide padding into 2 sides
    dh = pad_h // 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

    canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :] = img

    return canvas, r, dw, dh, new_unpad_w, new_unpad_h  # (dw,dh)

def _make_grid(nx=20, ny=20):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

def get_det_results(outputs,anchor_grid,stride,NUM_OUT):
    pred = []
    grid = [torch.zeros(1)] * 3

    for i in range(len(outputs)):
        bs, _ , ny, nx = outputs[i].shape
        # from n,c,h,w to n,3,85,h,w
        outputs[i]=outputs[i].view(bs, 3, NUM_OUT, ny*nx).permute(0, 1, 3, 2).view(bs, 3, ny, nx, NUM_OUT).contiguous()
        
        
        # grid[i], anchor_grid[i] = make_grid(anchors, nx, ny, i)
        if grid[i].shape[2:4] != outputs[i].shape[2:4]:
            grid[i] = _make_grid(nx, ny)
        
        y = outputs[i].sigmoid()
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid[i]) * stride[i]  # xy 计算预测框的中心坐标
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh

        pred.append(y.view(bs, -1, NUM_OUT))
    pred = torch.cat(pred, 1)  #torch.size([1, 25200, 85])
    return pred



def get_seg_results(da_seg_out,ll_seg_out,pre_messg):
    ratio, dw, dh, new_unpad_w, new_unpad_h,ori_shape = pre_messg
    # select da & ll segment area.
    da_seg_out = da_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]
    ll_seg_out = ll_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]

    da_seg_mask = np.argmax(da_seg_out, axis=1)[0]  # (?,?) (0|1)
    ll_seg_mask = np.argmax(ll_seg_out, axis=1)[0]  # (?,?) (0|1)

    return da_seg_mask, ll_seg_mask

def vis_det(ori_img,boxes,save_det_path,show,save):
    img_det = ori_img.copy()
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2, conf, label = boxes[i]
        x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
        img_det = cv2.rectangle(img_det, (x1, y1), (x2, y2), (255, 255, 0), 2, 2)
    if show:
        cv2.imshow("Traffic Object Detection Result", img_det)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if save:
        cv2.imwrite(save_det_path, img_det)


def vis_seg(da_seg_mask, ll_seg_mask, pre_messg, save_da_seg_path, save_ll_seg_path, show, save):
    width = pre_messg[-1][1]
    height = pre_messg[-1][0]
    # da: resize to original size
    da_seg_mask = da_seg_mask * 255
    da_seg_mask = da_seg_mask.numpy().astype(np.uint8)
    da_seg_mask = cv2.resize(da_seg_mask, (width, height),
                             interpolation=cv2.INTER_LINEAR)

    # ll: resize to original size
    ll_seg_mask = ll_seg_mask * 255
    ll_seg_mask = ll_seg_mask.numpy().astype(np.uint8)
    ll_seg_mask = cv2.resize(ll_seg_mask, (width, height),
                             interpolation=cv2.INTER_LINEAR)
    if show:
        cv2.imshow("Drivable Area Segmentation Result", da_seg_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow("Lane Detection Result", ll_seg_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if save:
        cv2.imwrite(save_da_seg_path, da_seg_mask)
        cv2.imwrite(save_ll_seg_path, ll_seg_mask)


def vis_merge(img, boxes, da_seg_mask, ll_seg_mask, pre_messg, save_merge_path, show, save):
    ratio, dw, dh, new_unpad_w, new_unpad_h, ori_shape = pre_messg
    width = pre_messg[-1][1]
    height = pre_messg[-1][0]

    color_area = np.zeros((new_unpad_h, new_unpad_w, 3), dtype=np.uint8)
    color_area[da_seg_mask == 1] = [0, 255, 0]
    color_area[ll_seg_mask == 1] = [255, 0, 0]
    color_seg = color_area

    # convert to BGR
    color_seg = color_seg[..., ::-1]
    color_mask = np.mean(color_seg, 2)
    img_merge = img[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :]
    img_merge = img_merge[:, :, ::-1]
    # merge: resize to original size
    img_merge[color_mask != 0] = \
        img_merge[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    img_merge = img_merge.astype(np.uint8)
    
    img_merge = cv2.resize(img_merge, (width, height),
                           interpolation=cv2.INTER_LINEAR)
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2, conf, label = boxes[i]
        x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
        img_merge = cv2.rectangle(img_merge, (x1, y1), (x2, y2), (255, 255, 0), 2, 2)
    if show:
        cv2.imshow("merge task", img_merge)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if save:
        cv2.imwrite(save_merge_path, img_merge)

    



def post_process_soft(ori_img,img,name,det_outs,da_seg_out,ll_seg_out,netinfo,anchors,nc,pre_messg,resRoot,show,save):
    strides = get_Stride(netinfo)
    save_det_path = resRoot + "/det_res_" + name
    save_da_seg_path = resRoot + "/da_seg_res_" + name
    save_ll_seg_path = resRoot + "/ll_seg_res_" + name
    save_merge_path = resRoot + "/merge_res_" + name
    # det postprocess
    nl = len(anchors)
    anchor_grid = torch.tensor(anchors).reshape((nl, 1, 3, 1, 1, 2))# shape(nl,1,na,1,1,2)
    pred = get_det_results(det_outs, anchor_grid, strides, nc+5)
    boxes = non_max_suppression(pred)[0]  # [n,6] [x1,y1,x2,y2,conf,cls]
    boxes = boxes.cpu().numpy().astype(np.float32)
    # scale coords to original size.
    boxes[:, 0] -= pre_messg[1] #dw
    boxes[:, 1] -= pre_messg[2] #dh
    boxes[:, 2] -= pre_messg[1]
    boxes[:, 3] -= pre_messg[2]
    boxes[:, :4] /= pre_messg[0] #ratio
    print(f"detect {boxes.shape[0]} bounding boxes.")

    # vis_det(ori_img, boxes, save_det_path, show, save)

    # da&ll  segment postprocess
    da_seg_mask, ll_seg_mask = get_seg_results(da_seg_out, ll_seg_out, pre_messg)

    # vis_seg(da_seg_mask, ll_seg_mask, pre_messg, save_da_seg_path, save_ll_seg_path, show, save)

    # merge tasks
    vis_merge(img, boxes, da_seg_mask, ll_seg_mask, pre_messg, save_merge_path, show, save)










