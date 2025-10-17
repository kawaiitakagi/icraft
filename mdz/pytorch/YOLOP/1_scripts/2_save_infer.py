import os
import cv2
import torch
import argparse
import onnxruntime as ort
import numpy as np
from lib.core.general import non_max_suppression

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

def make_grid(anchors,nx=20, ny=20, i=0):
    shape = 1, 3, ny, nx, 2  # grid shape
    y, x = torch.arange(ny), torch.arange(nx)
    yv, xv = torch.meshgrid(y, x)
    grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
    anchor_grid = anchors[i].view((1, 3, 1, 1, 2)).expand(shape)
    return grid, anchor_grid

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
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid[i]) * stride[i]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh

        pred.append(y.view(bs, -1, NUM_OUT))
    pred = torch.cat(pred, 1)  #torch.size([1, 25200, 85])
    return pred

def infer_yolop():

    ort.set_default_logger_severity(4)
    onnx_path = args.weight
    img_path = args.img
    ort_session = ort.InferenceSession(onnx_path)
    print(f"Load {onnx_path} done!")

    outputs_info = ort_session.get_outputs()
    inputs_info = ort_session.get_inputs()

    for ii in inputs_info:
        print("Input: ", ii)
    for oo in outputs_info:
        print("Output: ", oo)

    print("num outputs: ", len(outputs_info))

    save_det_path = args.resroot + "/detect_onnx.jpg"
    save_da_path = args.resroot + "/da_onnx.jpg"
    save_ll_path = args.resroot + "/ll_onnx.jpg"
    save_merge_path = args.resroot + "/output_onnx.jpg"

    img_bgr = cv2.imread(img_path)
    height, width, _ = img_bgr.shape

    # convert to RGB
    img_rgb = img_bgr[:, :, ::-1].copy()

    # resize & normalize
    canvas, r, dw, dh, new_unpad_w, new_unpad_h = resize_unscale(img_rgb, args.input_size)

    img = canvas.copy().astype(np.float32)  # (3,640,640) RGB
    # norm
    img /= 255.0
    img[:, :, 0] -= 0.485
    img[:, :, 1] -= 0.456
    img[:, :, 2] -= 0.406
    img[:, :, 0] /= 0.229
    img[:, :, 1] /= 0.224
    img[:, :, 2] /= 0.225

    img = img.transpose(2, 0, 1)

    img = np.expand_dims(img, 0)  # (1, 3,640,640)

    # inference: (1,n,6) (1,2,640,640) (1,2,640,640)    1, det_out2, det_out3
    det_out1, det_out2, det_out3, da_seg_out, ll_seg_out = ort_session.run(
        ['det_out1', 'det_out2','det_out3','drive_area_seg', 'lane_line_seg'],
        input_feed={"images": img}
    )

    det_outs = [torch.from_numpy(det_out1).float(),torch.from_numpy(det_out2).float(),torch.from_numpy(det_out3).float()]
    
    # det后处理
    NUM_CLASS = 1  # number of classes
    NUM_OUT = NUM_CLASS + 5  # number of outputs per anchor 85
    strides = [8,16,32]
    # anchors =[[[10.,13.],[16.,30.],[33.,23.]],[[30.,61.],[62.,45.],[59.,119.]],[[116.,90.],[156.,198.],[373.,326.]]] # yolov5
    anchors =[[[3.,9.],[5.,11.],[4.,20.]],[[7.,18.],[6.,39.],[12.,31.]],[[19.,50.],[38.,81.],[68.,157.]]] # yolop
    nl = len(anchors)
    anchor_grid = torch.tensor(anchors).reshape((nl, 1, 3, 1, 1, 2)) # shape(nl,1,na,1,1,2)
    pred = get_det_results(det_outs,anchor_grid,strides,NUM_OUT)

    boxes = non_max_suppression(pred)[0]  # [n,6] [x1,y1,x2,y2,conf,cls]
    boxes = boxes.cpu().numpy().astype(np.float32)

    if boxes.shape[0] == 0:
        print("no bounding boxes detected.")
        return

    # scale coords to original size.
    boxes[:, 0] -= dw
    boxes[:, 1] -= dh
    boxes[:, 2] -= dw
    boxes[:, 3] -= dh
    boxes[:, :4] /= r

    print(f"detect {boxes.shape[0]} bounding boxes.")

    img_det = img_rgb[:, :, ::-1].copy()
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2, conf, label = boxes[i]
        x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
        img_det = cv2.rectangle(img_det, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)

    cv2.imwrite(save_det_path, img_det)

    # select da & ll segment area.
    da_seg_out = da_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]
    ll_seg_out = ll_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]

    da_seg_mask = np.argmax(da_seg_out, axis=1)[0]  # (?,?) (0|1)
    ll_seg_mask = np.argmax(ll_seg_out, axis=1)[0]  # (?,?) (0|1)
    # print(da_seg_mask.shape)
    # print(ll_seg_mask.shape)

    color_area = np.zeros((new_unpad_h, new_unpad_w, 3), dtype=np.uint8)
    color_area[da_seg_mask == 1] = [0, 255, 0]
    color_area[ll_seg_mask == 1] = [255, 0, 0]
    color_seg = color_area

    # convert to BGR
    color_seg = color_seg[..., ::-1]
    color_mask = np.mean(color_seg, 2)
    img_merge = canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :]
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
        img_merge = cv2.rectangle(img_merge, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)

    # da: resize to original size
    da_seg_mask = da_seg_mask * 255
    da_seg_mask = da_seg_mask.astype(np.uint8)
    da_seg_mask = cv2.resize(da_seg_mask, (width, height),
                             interpolation=cv2.INTER_LINEAR)

    # ll: resize to original size
    ll_seg_mask = ll_seg_mask * 255
    ll_seg_mask = ll_seg_mask.astype(np.uint8)
    ll_seg_mask = cv2.resize(ll_seg_mask, (width, height),
                             interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(save_merge_path, img_merge)
    cv2.imwrite(save_da_path, da_seg_mask)
    cv2.imwrite(save_ll_path, ll_seg_mask)

    print("detect done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default="../2_compile/fmodel/yolop_1x3x640x640_traced.onnx")
    parser.add_argument('--img', type=str, default="./inference/images/9aa94005-ff1d4c9a.jpg")
    parser.add_argument('--resroot', type=str, default="./output/save_infer/")
    parser.add_argument('--input_size', type=list, default=[640,640]) # [h,w]
    args = parser.parse_args()

    if not os.path.exists(args.resroot):
        os.makedirs(args.resroot)

    infer_yolop()
    """
    PYTHONPATH=. python3 ./test_onnx.py --weight yolop-640-640.onnx --img test.jpg
    """
