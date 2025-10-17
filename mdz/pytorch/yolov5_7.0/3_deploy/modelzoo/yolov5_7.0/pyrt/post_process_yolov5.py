import sys
sys.path.append(R"../../../Deps/modelzoo")
from pyrtutils.utils import *
from pyrtutils.modelzoo_utils import *
from icraft.xir import *
from icraft.xrt import *
from icraft.buyibackend import *
from icraft.host_backend import *
import os
import time
import numpy as np
import cv2
import torch
import torchvision
from yolov5_utils import _COLORS,COCO_CLASSES
from yolov5_utils import *
# # Settings
# torch.set_printoptions(linewidth=320, precision=5, profile='long')
# np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
# cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
# os.environ['NUMEXPR_MAX_THREADS'] = str(min(os.cpu_count(), 8))  # NumExpr max threads


def make_grid(anchors,nx=20, ny=20, i=0):
    shape = 1, 3, ny, nx, 2  # grid shape
    y, x = torch.arange(ny), torch.arange(nx)
    
    yv, xv = torch.meshgrid(y, x)

    grid = torch.stack((xv, yv), 2)
    grid = grid.expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5

    anchor_grid = anchors[i].view((1, 3, 1, 1, 2)).expand(shape)
    return grid, anchor_grid

def xywh2xyxy(x):
    '''Convert boxes with shape [n, 4] from [x, y, w, h] to [x1, y1, x2, y2] where x1y1 is top-left, x2y2=bottom-right.'''
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results.
    This code is borrowed from: https://github.com/ultralytics/yolov5/blob/47233e1698b89fc437a4fb9463c815e9171be955/utils/general.py#L775
    Args:
        prediction: (tensor), with shape [N, 5 + num_classes], N is the number of bboxes.
        conf_thres: (float) confidence threshold.
        iou_thres: (float) iou threshold.
        classes: (None or list[int]), if a list is provided, nms only keep the classes you provide.
        agnostic: (bool), when it is set to True, we do class-independent nms, otherwise, different class would do nms respectively.
        multi_label: (bool), when it is set to True, one box can have multi labels, otherwise, one box only huave one label.
        max_det:(int), max number of output bboxes.

    Returns:
         list of detections, echo item is one tensor with shape (num_boxes, 6), 6 is for [xyxy, conf, cls].
    """

    num_classes = prediction.shape[2] - 5  # number of classes = 80
    pred_candidates = torch.logical_and(prediction[..., 4] > conf_thres, torch.max(prediction[..., 5:], axis=-1)[0] > conf_thres)  # candidates
    
    # Check the parameters.
    assert 0 <= conf_thres <= 1, f'conf_thresh must be in 0.0 to 1.0, however {conf_thres} is provided.'
    assert 0 <= iou_thres <= 1, f'iou_thres must be in 0.0 to 1.0, however {iou_thres} is provided.'

    # Function settings.
    max_wh = 4096  # maximum box width and height
    max_nms = 30000  # maximum number of boxes put into torchvision.ops.nms()
    time_limit = 10.0  # quit the function when nms cost time exceed the limit time.
    multi_label &= num_classes > 1  # multiple labels per box

    tik = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    
    for img_idx, x in enumerate(prediction):  # image index, image inference
        x = x[pred_candidates[img_idx]]  # confidence
        
        # If no box remains, skip the next process.
        if not x.shape[0]:
            continue

        # confidence multiply the objectness
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        
        # (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        # print('box =',box,box.shape)
        # exit()
        # Detections matrix's shape is  (n,6), each row represents (xyxy, conf, cls)
        if multi_label:
            box_idx, class_idx = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[box_idx], x[box_idx, class_idx + 5, None], class_idx[:, None].float()), 1)
        else:  # Only keep the class with highest scores.
            conf, class_idx = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, class_idx.float()), 1)[conf.view(-1) > conf_thres]
        
        # Filter by class, only keep boxes whose category is in classes.
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        num_box = x.shape[0]  # number of boxes
        if not num_box:  # no boxes kept.
            continue
        elif num_box > max_nms:  # excess max boxes' number.
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        class_offset = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + class_offset, x[:, 4]  # boxes (offset by class), scores
        keep_box_idx = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if keep_box_idx.shape[0] > max_det:  # limit detections
            keep_box_idx = keep_box_idx[:max_det]
        
        output[img_idx] = x[keep_box_idx]
        if (time.time() - tik) > time_limit:
            print(f'WARNING: NMS cost time exceed the limited {time_limit}s.')
            break  # time limit exceeded

    return output
def get_det_results_soft(generated_output,anchors,stride):
    pred = []
    grid = [torch.zeros(1)] * 3
    anchor_grid = [torch.zeros(1)] * 3
    outputs = []
    for i in range(len(generated_output)):
        output = torch.from_numpy(np.array(generated_output[i]).transpose(0,3,1,2))#模型中数据排布 from n,h,w,c to n,c,h,w 
        outputs.append(output)
    
    for i in range(len(outputs)):
        bs, _ , ny, nx = outputs[i].shape
        outputs[i] = outputs[i].view(bs, 3, 85, ny, nx).permute(0, 1, 3, 4, 2).contiguous() # from n,c,h,w to n,3,85,h,w
        
        grid[i], anchor_grid[i] = make_grid(anchors,nx, ny, i)
        
        y = outputs[i].sigmoid()
        
        y[..., 0:2] = (y[..., 0:2] * 2 + grid[i]) * stride[i]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
        
        pred.append(y.view(bs, -1, 85))
    pred = torch.cat(pred, 1)  #torch.size([1, 25200, 85])
    return pred


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
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
    
    if isinstance(boxes,np.ndarray):
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]],0,shape[1])  # x1, x2
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]],0,shape[0]) # y1, y2
    else:
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, shape[0])  # y1, y2

def vis(img, boxes, scores, cls_ids, conf=0.25, class_names=None):

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

def get_Stride(netinfo):
    strides = []
    ih = netinfo.i_cubic[0].h
    for i in netinfo.o_cubic:
        strides.append(ih // i.h)
        if ih % i.h:
            mprint("stride is not int", VERBOSE, 1)
    return strides


def post_detpost_soft(output_tensors,img_shape,img_raw,conf,netinfo,anchors):
    strides = get_Stride(netinfo)

    # ---------------------------------后处理---------------------------------
    #  from sim results to det results
    pred = get_det_results_soft(output_tensors,torch.Tensor(anchors),strides)
    # print('pred =',pred,pred.shape)
    # print("INFO: pred shape =",pred.shape)
    # NMS
    pred = non_max_suppression(pred, classes=None, agnostic=False, max_det=1000)

    # ---------------------------------结果可视化---------------------------------
    det = pred[0]
    det[:, :4] = scale_coords(img_shape[1:], det[:, :4], img_raw.shape).round()
    result_image = vis(img_raw, boxes=det[:,:4], scores=det[:,4], cls_ids=det[:,5], conf=conf, class_names=COCO_CLASSES)
    cv2.imshow(" ", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return




def get_det_results_hard(generated_output,scale_list,ANCHOR_LENGTH,ANCHORS,STRIDE ,N_CLASS,BIT):
    id_list = []
    scores_list = []
    box_list = []
    icore_post_res = []
    # flatten icore_post_result
    for i in range(len(generated_output)):
        output = np.array(generated_output[i]).flatten()#模型中数据排布 e.g [1,1,133,96] ->[133*96]
        icore_post_res.append(output)


    for i in range(len(icore_post_res)):
        objnum = icore_post_res[i].shape[0] / ANCHOR_LENGTH
        tensor_data = icore_post_res[i]
        
        for j in range(int(objnum)):
            obj_ptr_start = j * ANCHOR_LENGTH
            obj_ptr_next = obj_ptr_start + ANCHOR_LENGTH
            if BIT==16:
                anchor_index = tensor_data[obj_ptr_next - 1]
                location_y = tensor_data[obj_ptr_next - 2]
                location_x = tensor_data[obj_ptr_next - 3]
            elif BIT==8:
                anchor_index1 = tensor_data[j * ANCHOR_LENGTH + ANCHOR_LENGTH - 1]
                anchor_index2 = tensor_data[j * ANCHOR_LENGTH + ANCHOR_LENGTH - 2]
                location_y1 = tensor_data[j * ANCHOR_LENGTH + ANCHOR_LENGTH - 3]
                location_y2 = tensor_data[j * ANCHOR_LENGTH + ANCHOR_LENGTH - 4]
                location_x1 = tensor_data[j * ANCHOR_LENGTH + ANCHOR_LENGTH - 5]
                location_x2 = tensor_data[j * ANCHOR_LENGTH + ANCHOR_LENGTH - 6]
                anchor_index = (anchor_index1 << 8) + anchor_index2
                location_y = (location_y1 << 8) + location_y2
                location_x = (location_x1 << 8) + location_x2

            _x = sigmoid(tensor_data[obj_ptr_start ]    * scale_list[i])
            _y = sigmoid(tensor_data[obj_ptr_start + 1] * scale_list[i])
            _w = sigmoid(tensor_data[obj_ptr_start + 2] * scale_list[i])
            _h = sigmoid(tensor_data[obj_ptr_start + 3] * scale_list[i])
            _s = sigmoid(tensor_data[obj_ptr_start + 4] * scale_list[i])
            print(_x,_y,_w,_h,_s)
            
            class_ptr_start = obj_ptr_start + 5
            class_data_list = tensor_data[obj_ptr_start + 5:obj_ptr_start +5+N_CLASS]
            max_value = max(class_data_list)
            max_idx = list(class_data_list).index(max_value)
            realscore = _s / (1 + np.exp( - max_value * scale_list[i ]))

            x = (2*_x + location_x-0.5) * STRIDE[i]
            y = (2*_y + location_y-0.5) * STRIDE[i]
            w = 4 * (_w)**2  * ANCHORS[i][anchor_index][0]
            h = 4 * (_h)**2  * ANCHORS[i][anchor_index][1]

            scores_list.append(realscore)
            box_list.append(((x - w / 2), (y - h / 2), w, h))
            id_list.append(max_idx)
    return scores_list,box_list,id_list

def post_detpost_hard(output_tensors,img_shape,img_raw,img_path,netinfo,anchors,N_CLASS,conf,iou_thresh):

    strides = get_Stride(netinfo)
    generated_output = []
    for tensor in output_tensors:
        generated_output.append(np.asarray(tensor.to(HostDevice.MemRegion())))#data transfer2 [udma->ps]  
    scores_list,box_list,id_list = get_det_results_hard(generated_output,netinfo.o_scale,np.array(generated_output[0]).shape[3],anchors,strides ,N_CLASS,netinfo.detpost_bit)
    nms_indices,nms_box_list,nms_score_list,nms_cls_ids = soft_nms(box_list, scores_list, id_list,conf,iou_thresh,N_CLASS)
    nms_box_list = scale_coords(img_shape[1:], np.array(nms_box_list), img_raw.shape)
    result_image = vis(img_raw,boxes=nms_box_list,scores=nms_score_list,cls_ids=nms_cls_ids,conf=0.25,class_names=COCO_CLASSES)
    output_path = img_path.replace('input','output').replace('.jpg','_result.jpg')
    result_path = output_path[:output_path.rfind("\\")]
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    else:
        pass
    print("\n",'result save at',output_path)

    cv2.imwrite(output_path,result_image)

    return