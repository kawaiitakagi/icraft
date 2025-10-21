"""Python implementation of the YOLO11 post-processing routines.

This module mirrors the behaviour of the C++ utilities located in
``postprocess_yolo11.hpp`` so that the same deployment pipeline can be used
from Python scripts. Only the functionality required by
``post_detpost_hard`` is ported, which includes helper routines for stride
calculation, grid decoding and bounding box extraction.
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

try:  # Optional dependency – only available when running on target devices.
    from icraft.xrt import HostDevice  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    HostDevice = None  # type: ignore

try:  # FPGA accelerated NMS helper (optional).
    from ...Deps.modelzoo.pyrtutils.detpost_hard import fpga_nms  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    fpga_nms = None  # type: ignore


@dataclass
class Grid:
    """Decoded spatial information for a detection anchor."""

    location_x: int = 0
    location_y: int = 0
    anchor_index: int = 0


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def sigmoid(x: float) -> float:
    """Numerically stable sigmoid implementation."""

    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _softmax(x: np.ndarray) -> np.ndarray:
    """Softmax that guards against overflow."""

    x_max = np.max(x, axis=1, keepdims=True)
    exps = np.exp(x - x_max)
    return exps / np.sum(exps, axis=1, keepdims=True)


def dfl(data: Sequence[float], alpha: float, start: int, info_length: int) -> List[float]:
    """Distribution Focal Loss integral used by YOLOv8/YOLO11 heads.

    The input describes 4 independent distributions (left, top, right,
    bottom).  The function converts them to box edge offsets measured in
    pixels.
    """

    block = np.array(data[start : start + info_length], dtype=np.float32) * alpha
    block = block.reshape(4, info_length // 4)
    weights = _softmax(block)
    bins = np.arange(block.shape[1], dtype=np.float32)
    return (weights * bins[None, :]).sum(axis=1).tolist()


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def get_stride(netinfo) -> List[float]:
    """Compute the stride of each detection head."""

    input_height = netinfo.i_cubic[0].h
    strides: List[float] = []
    for cubic in netinfo.o_cubic:
        if cubic.h == 0:
            raise ValueError("Output feature-map height must be positive")
        stride = input_height / float(cubic.h)
        strides.append(stride)
    return strides


def get_grid(bits: int, tensor_data: Sequence[int], base_addr: int, anchor_length: int) -> Grid:
    """Decode the spatial grid index stored at the end of an anchor record."""

    if bits not in (8, 16):
        raise ValueError("Only 8-bit and 16-bit detpost outputs are supported")

    if bits == 8:
        anchor_index = (
            (int(tensor_data[base_addr + anchor_length - 1]) & 0xFF) << 8
        ) | (int(tensor_data[base_addr + anchor_length - 2]) & 0xFF)
        location_y = (
            (int(tensor_data[base_addr + anchor_length - 3]) & 0xFF) << 8
        ) | (int(tensor_data[base_addr + anchor_length - 4]) & 0xFF)
        location_x = (
            (int(tensor_data[base_addr + anchor_length - 5]) & 0xFF) << 8
        ) | (int(tensor_data[base_addr + anchor_length - 6]) & 0xFF)
    else:  # bits == 16
        anchor_index = int(tensor_data[base_addr + anchor_length - 1])
        location_y = int(tensor_data[base_addr + anchor_length - 2])
        location_x = int(tensor_data[base_addr + anchor_length - 3])

    return Grid(location_x=location_x, location_y=location_y, anchor_index=anchor_index)


def _tensor_to_numpy(tensor) -> np.ndarray:
    """Convert an icraft Tensor (or array-like) into an ``np.ndarray``."""

    if hasattr(tensor, "to") and HostDevice is not None:
        host_tensor = tensor.to(HostDevice.MemRegion())  # type: ignore[attr-defined]
        return np.asarray(host_tensor)
    return np.asarray(tensor)


def _extract_tensor_info(tensor) -> Tuple[np.ndarray, int, int, int]:
    """Return flattened tensor data along with bit width metadata."""

    array = _tensor_to_numpy(tensor)
    if array.ndim < 2:
        raise ValueError("Detpost tensor must be at least 2-D")
    obj_num = array.shape[-2]
    anchor_length = array.shape[-1]
    flat = np.ascontiguousarray(array).reshape(-1)
    bits = int(array.dtype.itemsize * 8)
    if bits not in (8, 16):
        raise ValueError(f"Unsupported detpost bit width: {bits}")
    return flat, bits, obj_num, anchor_length


# ---------------------------------------------------------------------------
# Detection decoding
# ---------------------------------------------------------------------------

def post_process(
    id_list: List[int],
    score_list: List[float],
    box_list: List[Tuple[float, float, float, float]],
    tensor_data: Sequence[int],
    obj_ptr_start: int,
    grid: Grid,
    real_out_channels: Sequence[int],
    bbox_info_channel: int,
    norm: Sequence[float],
    stride: float,
    anchors: Sequence[float],
    num_classes: int,
    conf_thresh: float,
    multi_label: bool,
) -> None:
    """Decode a single anchor record and append detection results."""

    del anchors  # Anchor handling is not required for YOLO11 detpost outputs.

    class_slice = [float(x) for x in tensor_data[obj_ptr_start : obj_ptr_start + num_classes]]
    if not class_slice:
        return

    if not multi_label:
        max_index = int(np.argmax(class_slice))
        prob = sigmoid(class_slice[max_index] * float(norm[0]))
        if prob <= conf_thresh:
            return
        offsets = dfl(
            tensor_data,
            float(norm[1]),
            obj_ptr_start + int(real_out_channels[0]),
            bbox_info_channel,
        )
        _append_detection(id_list, score_list, box_list, grid, offsets, stride, max_index, prob)
        return

    for class_idx, logit in enumerate(class_slice[:num_classes]):
        prob = sigmoid(logit * float(norm[0]))
        if prob <= conf_thresh:
            continue
        offsets = dfl(
            tensor_data,
            float(norm[1]),
            obj_ptr_start + int(real_out_channels[0]),
            bbox_info_channel,
        )
        _append_detection(id_list, score_list, box_list, grid, offsets, stride, class_idx, prob)


def _append_detection(
    id_list: List[int],
    score_list: List[float],
    box_list: List[Tuple[float, float, float, float]],
    grid: Grid,
    offsets: Sequence[float],
    stride: float,
    class_idx: int,
    prob: float,
) -> None:
    """Append a decoded detection to the running buffers."""

    x1 = grid.location_x + 0.5 - offsets[0]
    y1 = grid.location_y + 0.5 - offsets[1]
    x2 = grid.location_x + 0.5 + offsets[2]
    y2 = grid.location_y + 0.5 + offsets[3]

    cx = ((x2 + x1) / 2.0) * stride
    cy = ((y2 + y1) / 2.0) * stride
    w = (x2 - x1) * stride
    h = (y2 - y1) * stride

    box_list.append((cx - w / 2.0, cy - h / 2.0, w, h))
    id_list.append(class_idx)
    score_list.append(prob)


# ---------------------------------------------------------------------------
# Post-NMS utilities
# ---------------------------------------------------------------------------

def jaccard_distance(box1: Sequence[float], box2: Sequence[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2]) * max(0.0, box1[3])
    area2 = max(0.0, box2[2]) * max(0.0, box2[3])
    union = area1 + area2 - intersection
    if union <= 0:
        return 1.0
    return 1.0 - intersection / union


def nms_soft(
    id_list: Sequence[int],
    score_list: Sequence[float],
    box_list: Sequence[Tuple[float, float, float, float]],
    iou_thresh: float,
    max_nms: int = 3000,
) -> List[Tuple[int, float, Tuple[float, float, float, float]]]:
    """Class-wise Non-Maximum Suppression identical to the C++ helper."""

    candidates = [
        (cls_id, score, box)
        for cls_id, score, box in zip(id_list, score_list, box_list)
    ]
    candidates.sort(key=lambda item: item[1], reverse=True)

    nms_res: List[Tuple[int, float, Tuple[float, float, float, float]]] = []
    for cls_id, score, box in candidates:
        keep = True
        for kept_cls, _, kept_box in nms_res:
            if kept_cls == cls_id and (1.0 - jaccard_distance(box, kept_box)) > iou_thresh:
                keep = False
                break
        if keep:
            nms_res.append((cls_id, score, box))
        if len(nms_res) >= max_nms:
            break
    return nms_res


def nms_hard(
    box_list: Sequence[Tuple[float, float, float, float]],
    score_list: Sequence[float],
    id_list: Sequence[int],
    conf_thresh: float,
    iou_thresh: float,
    num_classes: int,
    device,
) -> List[Tuple[int, float, Tuple[float, float, float, float]]]:
    """Hardware accelerated NMS backed by ``fpga_nms`` when available."""

    if fpga_nms is None:
        raise NotImplementedError("Hardware NMS is not available in this environment")

    boxes_xyxy = [
        [b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in box_list
    ]
    selected_boxes = fpga_nms(
        boxes_xyxy, list(score_list), list(id_list), conf_thresh, iou_thresh, num_classes, device
    )
    selected: List[Tuple[int, float, Tuple[float, float, float, float]]] = []
    for sel in selected_boxes:
        for idx, candidate in enumerate(boxes_xyxy):
            if np.allclose(sel, candidate):
                selected.append((id_list[idx], score_list[idx], box_list[idx]))
                break
    return selected


def coord_trans(
    nms_res: Sequence[Tuple[int, float, Tuple[float, float, float, float]]],
    img,
    check_border: bool = True,
) -> List[List[float]]:
    """Transform coordinates back to the original image size."""

    output_data: List[List[float]] = []
    left_pad, top_pad = img.getPad()
    ratio = img.getRatio()[0]
    src_h, src_w = img.src_img.shape[:2]

    for cls_id, score, box in nms_res:
        x1 = (box[0] - left_pad) / ratio
        y1 = (box[1] - top_pad) / ratio
        x2 = (box[0] + box[2] - left_pad) / ratio
        y2 = (box[1] + box[3] - top_pad) / ratio

        if check_border:
            x1 = min(max(x1, 0.0), float(src_w))
            y1 = min(max(y1, 0.0), float(src_h))
            x2 = min(max(x2, 0.0), float(src_w))
            y2 = min(max(y2, 0.0), float(src_h))

        output_data.append([float(cls_id), x1, y1, x2 - x1, y2 - y1, score])
    return output_data


def visualize(
    output_res: Sequence[Sequence[float]],
    image: np.ndarray,
    res_root: str,
    name: str,
    labels: Sequence[str],
    show: bool = False,
    save: bool = False,
) -> None:
    """Visualise detections on the image and optionally display/save them."""

    if image is None:
        return

    canvas = image.copy()
    rng = random.Random(0)
    for res in output_res:
        class_id = int(res[0])
        x1, y1, w, h, score = map(float, res[1:6])
        color = tuple(int(rng.randint(10, 200)) for _ in range(3))
        top_left = (int(round(x1)), int(round(y1)))
        bottom_right = (int(round(x1 + w)), int(round(y1 + h)))
        cv2.rectangle(canvas, top_left, bottom_right, color, 2)
        if 0 <= class_id < len(labels):
            label = labels[class_id]
        else:
            label = str(class_id)
        text = f"{class_id}_{label} {score:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
        cv2.rectangle(
            canvas,
            (top_left[0] - 1, top_left[1] - text_height - 7),
            (top_left[0] + text_width, top_left[1] - 2),
            color,
            thickness=-1,
        )
        cv2.putText(
            canvas,
            text,
            (top_left[0], top_left[1] - 2),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    if save:
        os.makedirs(res_root, exist_ok=True)
        base, ext = os.path.splitext(name)
        save_name = f"{base}_result{ext}" if ext else f"{name}_result"
        save_path = os.path.join(res_root, save_name)
        cv2.imwrite(save_path, canvas)
    if show:
        cv2.imshow("results", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def save_res(output_res: Sequence[Sequence[float]], res_root: str, name: str) -> None:
    """Persist raw detection data to a text file."""

    os.makedirs(res_root, exist_ok=True)
    base, _ = os.path.splitext(name)
    save_path = os.path.join(res_root, f"{base}.txt")
    with open(save_path, "w", encoding="utf-8") as handle:
        for record in output_res:
            handle.write(" ".join(str(v) for v in record))
            handle.write("\n")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def post_detpost_hard(
    output_tensors: Sequence,
    img,
    netinfo,
    conf: float,
    iou_thresh: float,
    multi_label: bool,
    fpga_nms: bool,
    num_classes: int,
    anchors: Sequence[Sequence[Sequence[float]]],
    labels: Sequence[str],
    show: bool,
    save: bool,
    res_root: str,
    name: str,
    device,
    run_sim: bool,
    norm: Sequence[Sequence[float]],
    real_out_channels: Sequence[int],
    stride_list: Sequence[float],
    bbox_info_channel: int,
) -> None:
    """Python translation of ``post_detpost_hard`` from the C++ deployment kit."""

    del anchors  # Anchors are not required because detpost outputs are grid based.

    id_list: List[int] = []
    score_list: List[float] = []
    box_list: List[Tuple[float, float, float, float]] = []

    for head_index, tensor in enumerate(output_tensors):
        tensor_data, bits, obj_num, anchor_length = _extract_tensor_info(tensor)
        norm_head = norm[head_index]
        stride = float(stride_list[head_index])

        for obj in range(obj_num):
            base_addr = obj * anchor_length
            grid = get_grid(bits, tensor_data, base_addr, anchor_length)
            post_process(
                id_list,
                score_list,
                box_list,
                tensor_data,
                base_addr,
                grid,
                real_out_channels,
                bbox_info_channel,
                norm_head,
                stride,
                [],
                num_classes,
                conf,
                multi_label,
            )

    if not id_list:
        detections: List[Tuple[int, float, Tuple[float, float, float, float]]] = []
    elif fpga_nms and not run_sim:
        try:
            detections = nms_hard(box_list, score_list, id_list, conf, iou_thresh, num_classes, device)
        except NotImplementedError:
            detections = nms_soft(id_list, score_list, box_list, iou_thresh)
    else:
        detections = nms_soft(id_list, score_list, box_list, iou_thresh)

    output_res = coord_trans(detections, img)

    if os.name == "nt":
        if show:
            visualize(output_res, img.ori_img, res_root, name, labels, show=True, save=False)
        if save:
            save_res(output_res, res_root, name)
    else:
        if save:
            visualize(output_res, img.ori_img, res_root, name, labels, show=False, save=True)

