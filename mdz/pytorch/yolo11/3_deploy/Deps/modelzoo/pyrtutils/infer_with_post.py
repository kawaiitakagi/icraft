"""Minimal example showing how to run inference with the icraft Python runtime.

The script loads a compiled model (JSON + RAW files), prepares a single image, and
executes one forward pass. It intentionally keeps the preprocessing lightweight so the
example stays focused on the runtime APIs exposed in ``pyrtutils``.
"""

import argparse
import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm
import time
import json

# Allow ``import pyrtutils`` when the script lives inside the utilities package.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from pyrtutils.icraft_utils import (  # noqa: E402  pylint: disable=wrong-import-position
    getJrPath,
    initSession,
    loadNetwork,
    numpy2Tensor,
    openDevice,
)
from pyrtutils.Netinfo import Netinfo  # noqa: E402  pylint: disable=wrong-import-position
from pyrtutils.et_device import dmaInit  # noqa: E402  pylint: disable=wrong-import-position
from pyrtutils.modelzoo_utils import soft_nms  # noqa: E402  pylint: disable=wrong-import-position
from pyrtutils.utils import VERBOSE, mprint  # noqa: E402  pylint: disable=wrong-import-position

# from icraft.host_backend import HostDevice  # noqa: E402  pylint: disable=wrong-import-position
# from icraft.xrt import Device  # noqa: E402  pylint: disable=wrong-import-position

from icraft.xir import *
from icraft.xrt import *
from icraft.buyibackend import *
from icraft.host_backend import *
from pyrtutils.hard_utils import (  # noqa: E402  pylint: disable=wrong-import-position
    # VERBOSE,
    # dmaInit as hardware_dma_init,
    fpgaOPlist,
    getOutputNormratio,
    # mprint,
)
def _sigmoid(x: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64, copy=False)
    return 1.0 / (1.0 + np.exp(-x64))
# print(VERBOSE)

def _dfl(dist: np.ndarray, value_range: np.ndarray) -> np.ndarray:
    reshaped = dist.reshape(4, -1).astype(np.float64, copy=False)
    logits = reshaped - reshaped.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return (probs * value_range).sum(axis=1)

# def getOutputNormratio(network):
#     # 从xir.network获取输出的norm_ratio用于反量化
#     net_out_results = network.outputs()
#     scale_list = []
#     for value in net_out_results:
#         scale = value.dtype.getNormratio().data
#         scale_list.append(scale[0])
#     return scale_list

# def get_scale(GENERATED_JSON_FILE):
#     # 读取json文件获取反量化系数
#     with open(GENERATED_JSON_FILE,'r') as f:
#         net = json.load(f)
#     scale_list = []
#     # 从json文件中获取输出的 norm_ratio 用于反量化
#     for ftmp in net["ops"][-2]["inputs"]:
#         scale_list.append(ftmp["dtype"]["element_dtype"]["normratio"][0]["value"])
#     return scale_list

def _decode_detections(
    outputs: List[np.ndarray],
    netinfo: Netinfo,
    conf_thresh: float,
    iou_thresh: float,
    num_classes: Optional[int],
    multi_label: bool,
    meta: dict,
) -> List[List[float]]:
    """Decode model outputs to detection boxes, correcting for letterbox scaling."""
    if len(outputs) % 2 != 0:
        raise RuntimeError("Expected detection outputs in class/box pairs.")

    input_h = netinfo.i_cubic[0].h
    input_w = netinfo.i_cubic[0].w
    scale = meta["scale"]
    pad_x, pad_y = meta["pad"]
    orig_h, orig_w = meta["original_hw"]

    all_boxes, all_scores, all_ids = [], [], []

    for head in range(0, len(outputs), 2):
        cls_pred = np.asarray(outputs[head]).squeeze()
        box_pred = np.asarray(outputs[head + 1]).squeeze()

        if cls_pred.ndim != 3 or box_pred.ndim != 3:
            # print(cls_pred, box_pred)
            # print(cls_pred.shape, box_pred.shape)
            raise RuntimeError("Unexpected tensor layout. Expected [H, W, C].")

        height, width, channels = cls_pred.shape
        head_num_classes = channels if num_classes is None else num_classes

        stride_y = input_h / float(height)
        stride_x = input_w / float(width)

        reg_max = box_pred.shape[-1] // 4
        value_range = np.arange(reg_max, dtype=np.float64)

        for y in range(height):
            for x in range(width):
                class_logits = cls_pred[y, x, :head_num_classes]
                class_scores = 1.0 / (1.0 + np.exp(-class_logits))  # sigmoid

                if not multi_label:
                    cls_idx = int(np.argmax(class_scores))
                    score = float(class_scores[cls_idx])
                    if score < conf_thresh:
                        continue
                    distances = _dfl(box_pred[y, x], value_range)
                    x1 = (x + 0.5 - distances[0]) * stride_x
                    y1 = (y + 0.5 - distances[1]) * stride_y
                    x2 = (x + 0.5 + distances[2]) * stride_x
                    y2 = (y + 0.5 + distances[3]) * stride_y
                    all_boxes.append([x1, y1, x2 - x1, y2 - y1])
                    all_scores.append(score)
                    all_ids.append(cls_idx)
                else:
                    valid_indices = np.where(class_scores >= conf_thresh)[0]
                    if valid_indices.size == 0:
                        continue
                    distances = _dfl(box_pred[y, x], value_range)
                    x1 = (x + 0.5 - distances[0]) * stride_x
                    y1 = (y + 0.5 - distances[1]) * stride_y
                    x2 = (x + 0.5 + distances[2]) * stride_x
                    y2 = (y + 0.5 + distances[3]) * stride_y
                    for cls_idx in valid_indices:
                        all_boxes.append([x1, y1, x2 - x1, y2 - y1])
                        all_scores.append(float(class_scores[cls_idx]))
                        all_ids.append(int(cls_idx))

    if not all_boxes:
        return []

    classes = num_classes or int(max(all_ids) + 1)
    _, nms_boxes, nms_scores, nms_ids = soft_nms(
        all_boxes, all_scores, all_ids, conf=conf_thresh, iou=iou_thresh, NOC=classes
    )

    detections = []
    for cls_idx, score, box in zip(nms_ids, nms_scores, nms_boxes):
        # 还原到 letterbox 前坐标系
        x1 = (box[0] - pad_x) / scale
        y1 = (box[1] - pad_y) / scale
        x2 = (box[2] - pad_x) / scale
        y2 = (box[3] - pad_y) / scale

        # 限制在原图范围内
        x1 = np.clip(x1, 0, orig_w - 1)
        y1 = np.clip(y1, 0, orig_h - 1)
        x2 = np.clip(x2, 0, orig_w - 1)
        y2 = np.clip(y2, 0, orig_h - 1)

        detections.append([int(cls_idx), float(score), float(x1), float(y1), float(x2), float(y2)])

    detections.sort(key=lambda item: item[1], reverse=True)
    return detections


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-image inference with icraft.")
    parser.add_argument(
        "--model-dir",
        # required=True,
        default=r"C:\Users\92032\Downloads\runtime_demo\ultralytics-16bit\imodel_hard",
        help="Directory that contains the compiled model (the JSON/RAW pair).",
    )
    parser.add_argument(
        "--stage",
        default="g",
        choices=["p", "o", "q", "a", "g"],
        help="Model stage to load when multiple versions exist under the model directory.",
    )
    parser.add_argument(
        "--image",
        # required=True,
        default=r"C:\Users\92032\Downloads\runtime_demo\ultralytics-16bit\doppler_dataset\images\val",
        help="Path to the image that will be used as input.",
    )
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Run in simulator mode (Host backend only, no hardware required).",
    )
    parser.add_argument(
        "--ip",
        default="192.168.0.10",
        help="Device IP address (ignored in simulation mode).",
    )
    parser.add_argument(
        "--enable-mmu",
        action="store_true",
        help="Force-enable MMU mode when opening the device.",
    )
    parser.add_argument(
        "--speedmode",
        action="store_true",
        help="Enable the model's speedmode optimization when creating the session.",
    )
    parser.add_argument(
        "--compress-ftmp",
        action="store_true",
        help="Enable feature compression when creating the session.",
    )
    parser.add_argument(
        "--cuda-sim",
        action="store_true",
        help="Use the CUDA simulator backend instead of the CPU simulator backend.",
    )
    parser.add_argument(
        "--npu-addr",
        default="0x40000000",
        help="Physical base address of the NPU (used when connecting to hardware).",
    )
    parser.add_argument(
        "--dma-addr",
        default="0x80000000",
        help="Physical base address of the DMA (used when connecting to hardware).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.1,
        help="Confidence threshold applied before Soft-NMS.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.2,
        help="IoU threshold for Soft-NMS suppression.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Override the number of classes reported by the network outputs.",
    )
    parser.add_argument(
        "--multi-label",
        action="store_true",
        help="Keep boxes for every class above the confidence threshold (multi-label mode).",
    )
    parser.add_argument(
        "--save-vis",
        default=r'.\output',
        help="Path where the visualization image with detection boxes will be stored.",
    )
    return parser.parse_args()

def _prepare_input_image(
    netinfo: Netinfo,
    image_path: str,
    stage: str,
    run_sim: bool,
):
    """Load image and apply YOLO-style letterbox to match model input size."""
    if '.tiff' in image_path.lower():
        ret, raw_image = cv2.imreadmulti(image_path, flags=cv2.IMREAD_UNCHANGED)
        raw_image = np.stack(raw_image, axis=-1)  # Shape: (H, W, num_pages)
        bottom_patch = raw_image[-30:, :, :]
        raw_image = np.concatenate([bottom_patch, raw_image], axis=0)
    else:
        raw_image = cv2.imread(image_path)
    if raw_image is None:
        raise FileNotFoundError(f"Unable to load image: {image_path}")

    if not netinfo.i_cubic:
        raise RuntimeError("The loaded network does not expose any image-like input tensors.")

    cubic = netinfo.i_cubic[0]
    target_w, target_h, target_c = cubic.w, cubic.h, cubic.c
    h0, w0 = raw_image.shape[:2]
    # print(netinfo.value)

    # ---- Letterbox resize ----
    scale = min(target_w / w0, target_h / h0)
    new_w, new_h = int(round(w0 * scale)), int(round(h0 * scale))
    resized = cv2.resize(raw_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 创建灰色画布
    canvas = np.full((target_h, target_w, cubic.c), 128, dtype=resized.dtype)
    dw, dh = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[dh:dh + new_h, dw:dw + new_w] = resized

    # 通道对齐
    if canvas.ndim == 2:
        canvas = canvas[:, :, None]
    if canvas.shape[2] != target_c:
        canvas = canvas[:, :, :target_c]

    data = canvas
    if stage not in ("a", "g") and run_sim:
        data = data.astype(np.float32)

    # 返回 letterbox 相关参数，后续用于反变换检测框
    meta = dict(
        scale=scale,
        pad=(dw, dh),
        original_hw=(h0, w0),
        input_hw=(target_h, target_w),
    )

    return data.reshape(netinfo.i_shape[0]), raw_image, meta


def _visualize_detections(
    image: np.ndarray,
    detections: List[List[float]],
    save_path: str,
) -> None:
    from PIL import Image
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    # canvas = image.copy()
    print("original image shape:",image.shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("image after cvtColor shape:",image.shape)
    canvas = (image / image.max() * 255).astype(np.uint8)
    print("canvas shape:",canvas.shape)
    # canvas = Image.fromarray(vis).convert('RGB') 
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
    cv2.line(canvas, (0,30), (1024,30), (255,0,0),1)

    for det in detections:
        cls_idx, score, x1, y1, x2, y2 = det
        x1_i = int(round(x1))
        y1_i = int(round(y1))
        x2_i = int(round(x2))
        y2_i = int(round(y2))
        cls_idx = int(cls_idx)

        x1_i = max(0, min(canvas.shape[1] - 1, x1_i))
        y1_i = max(0, min(canvas.shape[0] - 1, y1_i))
        x2_i = max(0, min(canvas.shape[1] - 1, x2_i))
        y2_i = max(0, min(canvas.shape[0] - 1, y2_i))

        if x2_i < x1_i:
            x1_i, x2_i = x2_i, x1_i
        if y2_i < y1_i:
            y1_i, y2_i = y2_i, y1_i

        cv2.rectangle(canvas, (x1_i, y1_i), (x2_i, y2_i), colors[cls_idx], 2)
        label = f"{cls_idx}:{score:.2f}"
        cv2.putText(
            canvas,
            label,
            (x1_i, max(0, y1_i - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colors[cls_idx],
            1,
            cv2.LINE_AA,
        )

    if not cv2.imwrite(save_path, canvas):
        raise IOError(f"Failed to write visualization image to {save_path}")
    mprint(f"Saved visualization to {save_path}", VERBOSE, 0)

def yolo_postprocess_keep_below30(detections, y_threshold=30):
    """
    YOLO 检测后处理：
    1. 丢弃检测框完全在 y < y_threshold 之上的框；
    2. 保留跨过 y_threshold 或在其以下的框；
    3. 若框跨过30像素线且 y2不超过35，则记录其横向中心点；
       在64像素线下方寻找同类框，若中心点接近则去除上方框。

    参数:
        detections (ndarray): shape (N, 6) or (N, >=6)
            每一行: [cls, conf, x1, y1, x2, y2]
        y_threshold (int): y轴阈值，默认30像素。

    返回:
        ndarray: 过滤后的检测结果。
    """
    if isinstance(detections, list):
        detections = np.array(detections)

    if detections.size == 0:
        return detections

    y1 = detections[:, 3]
    y2 = detections[:, 5]

    # 初步筛选：保留跨过或在 y_threshold 下方的框
    keep_mask = y2 >= y_threshold
    kept = detections[keep_mask]

    if kept.size == 0:
        return kept

    # -----------------------------
    # 追加规则处理
    # -----------------------------
    to_remove = set()

    for i, det in enumerate(kept):
        cls, conf, x1, y1, x2, y2 = det[:6]
        if y1 < 30 <= y2 <= 35:  # 跨30线且y2不超过35
            cx = (x1 + x2) / 2  # 横向中心
            # 查找在64像素线下方的同类框
            below_mask = (kept[:, 0] == cls) & (kept[:, 5] > 64)
            same_cls_below = kept[below_mask]

            for j, below_det in enumerate(same_cls_below):
                bx1, by1, bx2, by2 = below_det[2:6]
                bcx = (bx1 + bx2) / 2
                if abs(cx - bcx) < 5:  # 横向中心接近
                    to_remove.add(i)
                    break

    if to_remove:
        keep_indices = [i for i in range(len(kept)) if i not in to_remove]
        kept = kept[keep_indices]

    return kept

def get_det_results_hard(generated_output,scale_list,ANCHOR_LENGTH=ANCHOR_LENGTH,ANCHORS=ANCHORS,STRIDE = STRIDE,N_CLASS=NOC,BIT=BIT):
    id_list = []
    scores_list = []
    box_list = []
    icore_post_res = []
    # flatten icore_post_result
    for i in range(len(generated_output)):
        output = np.array(generated_output[i]).flatten()#模型中数据排布 e.g [1,1,133,96] ->[133*96]
        icore_post_res.append(output)

    print('INFO: get icore_post flatten results!')

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

def process_image(image_path: str, args, netinfo, session, device, network):
    """Process a single image and perform inference."""
    t1 = time.time()
    input_data, raw_image, meta = _prepare_input_image(netinfo, image_path, args.stage, args.sim)
    input_tensor = numpy2Tensor(input_data, network)

    if not args.sim:
        dmaInit(args.sim, netinfo.ImageMake_on, netinfo.i_shape[0][1:], input_tensor, device)
    
    t2 = time.time()
    print(f"Loading time for {image_path}: {t2 - t1:.3f} seconds")

    outputs = session.forward([input_tensor])
    
    device.reset(1)

    output_arrays: List[np.ndarray] = []
    for index, tensor in enumerate(outputs):
        try:
            array = np.asarray(tensor.to(HostDevice.MemRegion()))
        except Exception:
            array = np.asarray(tensor)
        output_arrays.append(array)

    conf, box, id = get_det_results_hard(output_arrays,getOutputNormratio(network),ANCHOR_LENGTH=1,ANCHORS=ANCHORS,STRIDE = STRIDE,N_CLASS=NOC,BIT=BIT)

    detections = _decode_detections(
        output_arrays, netinfo, args.conf, args.iou,
        args.num_classes, args.multi_label, meta,
    )

    detections = yolo_postprocess_keep_below30(detections, y_threshold=30).tolist()

    t3 = time.time()
    print(f"Inference time for {image_path}: {t3 - t2:.3f} seconds")

    if detections:
        print(f"Detections for {image_path}:")
        for det in detections:
            print(str(det))
    else:
        print(f"No detections above the confidence threshold for {image_path}.")

    # === 输出可视化结果 ===
    output_dir = args.save_vis
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(output_dir, f"{base_name}_det.png")
    _visualize_detections(raw_image, detections, save_path)
    print(f"Visualization saved to {save_path}")

    # === 保存检测框到txt ===
    if detections:
        h, w = raw_image.shape[:2]
        label_save_dir = os.path.join(output_dir, "labels")
        os.makedirs(label_save_dir, exist_ok=True)
        label_path = os.path.join(label_save_dir, f"{base_name}.txt")

        with open(label_path, "w") as f:
            for det in detections:
                # det 格式假设为 [cls_id, conf, x1, y1, x2, y2]
                cls_id, conf, x1, y1, x2, y2 = det
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                f.write(f"{int(cls_id)} {conf:.4f} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        print(f"Label file saved to {label_path}")


def batch_process_images(input_dir: str, args, netinfo, session, device, network):
    """Batch process all images in the directory."""
    for filename in tqdm(os.listdir(input_dir)):
        if filename.lower().endswith('.tiff'):
            image_path = os.path.join(input_dir, filename)
            process_image(image_path, args, netinfo, session, device, network)

def main() -> None:
    args = _parse_args()

    json_path, raw_path = getJrPath(args.model_dir, args.stage, args.sim)
    print(f"Using model files: {json_path} / {raw_path}")

    network = loadNetwork(json_path, raw_path)
    netinfo = Netinfo(network)
    network_view = network.view(netinfo.inp_shape_opid + 1)

    mmu_required = netinfo.mmu or args.enable_mmu

    if args.sim:
        device = HostDevice.Default()
    else:
        device = openDevice(
            False,
            args.ip,
            mmu_Mode=mmu_required,
            cuda_Mode=args.cuda_sim,
            npu_addr=args.npu_addr,
            dma_addr=args.dma_addr,
        )

    session = initSession(
        args.sim,
        network_view,
        device,
        mmu_required,
        open_speedmode=args.speedmode,
        open_compressFtmp=args.compress_ftmp,
    )
    session.apply()

    # Now we process the images in the provided directory
    batch_process_images(args.image, args, netinfo, session, device, network)

    if not args.sim:
        Device.Close(device)
    
    print("Batch inference completed successfully.")

if __name__ == "__main__":
    main()
