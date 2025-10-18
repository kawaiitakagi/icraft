"""Minimal example showing how to run inference with the icraft Python runtime.

The script loads a compiled model (JSON + RAW files), prepares a single image, and
executes one forward pass. It intentionally keeps the preprocessing lightweight so the
example stays focused on the runtime APIs exposed in ``pyrtutils``.
"""

import argparse
import os
import sys
from typing import List, Optional

import cv2
import numpy as np

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
from pyrtutils.modelzoo_utils import soft_nms as legacy_soft_nms  # noqa: E402  pylint: disable=wrong-import-position
from pyrtutils.utils import (  # noqa: E402  pylint: disable=wrong-import-position
    VERBOSE,
    dmaInit as hardware_dma_init,
    fpgaOPlist,
    getOutputNormratio,
    mprint,
)
from pyrtutils.detpost_hard import (  # noqa: E402  pylint: disable=wrong-import-position
    NOC as HARD_DEFAULT_NUM_CLASSES,
    get_det_results,
    soft_nms as hard_soft_nms,
)

# from icraft.host_backend import HostDevice  # noqa: E402  pylint: disable=wrong-import-position
# from icraft.xrt import Device  # noqa: E402  pylint: disable=wrong-import-position

from icraft.xir import *
from icraft.xrt import *
from icraft.buyibackend import *
from icraft.host_backend import *
def _sigmoid(x: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64, copy=False)
    return 1.0 / (1.0 + np.exp(-x64))


def _dfl(dist: np.ndarray, value_range: np.ndarray) -> np.ndarray:
    reshaped = dist.reshape(4, -1).astype(np.float64, copy=False)
    logits = reshaped - reshaped.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return (probs * value_range).sum(axis=1)


# def _decode_detections(
#     outputs: List[np.ndarray],
#     netinfo: Netinfo,
#     conf_thresh: float,
#     iou_thresh: float,
#     num_classes: Optional[int],
#     multi_label: bool,
#     original_hw: Tuple[int, int],
# ) -> List[List[float]]:
#     if len(outputs) % 2 != 0:
#         raise RuntimeError(
#             "The demo expects detection models that produce class and box tensors in pairs."
#         )

#     input_h = netinfo.i_cubic[0].h
#     input_w = netinfo.i_cubic[0].w
#     scale_y = original_hw[0] / float(input_h)
#     scale_x = original_hw[1] / float(input_w)

#     all_boxes: List[List[float]] = []
#     all_scores: List[float] = []
#     all_ids: List[int] = []

#     for head in range(0, len(outputs), 2):
#         cls_pred = np.asarray(outputs[head]).squeeze()
#         box_pred = np.asarray(outputs[head + 1]).squeeze()

#         if cls_pred.ndim != 3 or box_pred.ndim != 3:
#             raise RuntimeError(
#                 "Unexpected tensor layout. Expected [H, W, C] for each detection head."
#             )

#         height, width, channels = cls_pred.shape
#         head_num_classes = channels if num_classes is None else num_classes

#         stride_y = input_h / float(height)
#         stride_x = input_w / float(width)

#         reg_max = box_pred.shape[-1] // 4
#         value_range = np.arange(reg_max, dtype=np.float64)

#         for y in range(height):
#             for x in range(width):
#                 class_logits = cls_pred[y, x, :head_num_classes]
#                 class_scores = _sigmoid(class_logits)

#                 if not multi_label:
#                     cls_idx = int(np.argmax(class_scores))
#                     score = float(class_scores[cls_idx])
#                     if score < conf_thresh:
#                         continue
#                     distances = _dfl(box_pred[y, x], value_range)
#                     x1 = (x + 0.5 - distances[0]) * stride_x
#                     y1 = (y + 0.5 - distances[1]) * stride_y
#                     x2 = (x + 0.5 + distances[2]) * stride_x
#                     y2 = (y + 0.5 + distances[3]) * stride_y
#                     all_boxes.append([x1, y1, x2 - x1, y2 - y1])
#                     all_scores.append(score)
#                     all_ids.append(cls_idx)
#                 else:
#                     valid_indices = np.where(class_scores >= conf_thresh)[0]
#                     if valid_indices.size == 0:
#                         continue
#                     distances = _dfl(box_pred[y, x], value_range)
#                     x1 = (x + 0.5 - distances[0]) * stride_x
#                     y1 = (y + 0.5 - distances[1]) * stride_y
#                     x2 = (x + 0.5 + distances[2]) * stride_x
#                     y2 = (y + 0.5 + distances[3]) * stride_y
#                     for cls_idx in valid_indices:
#                         all_boxes.append([x1, y1, x2 - x1, y2 - y1])
#                         all_scores.append(float(class_scores[cls_idx]))
#                         all_ids.append(int(cls_idx))

#     if not all_boxes:
#         return []

#     classes = num_classes or int(max(all_ids) + 1)
#     _, nms_boxes, nms_scores, nms_ids = soft_nms(
#         all_boxes,
#         all_scores,
#         all_ids,
#         conf=conf_thresh,
#         iou=iou_thresh,
#         NOC=classes,
#     )

#     detections = []
#     for cls_idx, score, box in zip(nms_ids, nms_scores, nms_boxes):
#         x1 = box[0] * scale_x
#         y1 = box[1] * scale_y
#         x2 = box[2] * scale_x
#         y2 = box[3] * scale_y
#         detections.append([int(cls_idx), float(score), float(x1), float(y1), float(x2), float(y2)])

#     detections.sort(key=lambda item: item[1], reverse=True)
#     return detections
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
    scale = meta["scale"] or 1.0
    pad_x, pad_y = meta["pad"]
    orig_h, orig_w = meta["original_hw"]

    all_boxes, all_scores, all_ids = [], [], []

    for head in range(0, len(outputs), 2):
        cls_pred = np.asarray(outputs[head]).squeeze()
        box_pred = np.asarray(outputs[head + 1]).squeeze()

        if cls_pred.ndim != 3 or box_pred.ndim != 3:
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
    _, nms_boxes, nms_scores, nms_ids = legacy_soft_nms(
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


def _decode_detections_hard(
    outputs: List[np.ndarray],
    network,
    meta: dict,
    conf_thresh: float,
    iou_thresh: float,
    num_classes: Optional[int],
) -> List[List[float]]:
    """Decode detection tensors that follow the hardware post-process layout."""

    if not outputs:
        return []

    scale_list = getOutputNormratio(network)
    classes = num_classes or HARD_DEFAULT_NUM_CLASSES

    scores, boxes, ids = get_det_results(outputs, scale_list, N_CLASS=classes)
    if not boxes:
        return []

    _, nms_boxes, nms_scores, nms_ids = hard_soft_nms(
        boxes,
        scores,
        ids,
        conf=conf_thresh,
        iou=iou_thresh,
        NOC=classes,
    )

    if not nms_boxes:
        return []

    scale = meta["scale"] or 1.0
    pad_x, pad_y = meta["pad"]
    orig_h, orig_w = meta["original_hw"]

    detections: List[List[float]] = []
    for cls_idx, score, box in zip(nms_ids, nms_scores, nms_boxes):
        x1 = (box[0] - pad_x) / scale
        y1 = (box[1] - pad_y) / scale
        x2 = (box[2] - pad_x) / scale
        y2 = (box[3] - pad_y) / scale

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
        required=True,
        help="Directory that contains the compiled model (the JSON/RAW pair).",
    )
    parser.add_argument(
        "--stage",
        default="a",
        choices=["p", "o", "q", "a", "g"],
        help="Model stage to load when multiple versions exist under the model directory.",
    )
    parser.add_argument(
        "--image",
        # required=True,
        default=r"C:\Users\92032\Downloads\runtime_demo\examples\yolo_sim_demo\yolo11l\io\input\1_0312.png",
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
        default=0.3,
        help="Confidence threshold applied before Soft-NMS.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.75,
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
        "--hard-layout",
        action="store_true",
        help="Enable hardware post-process decoding for custom operator layouts.",
    )
    parser.add_argument(
        "--save-vis",
        default=r'.\output\test.png',
        help="Path where the visualization image with detection boxes will be stored.",
    )
    return parser.parse_args()


# def _prepare_input_image(
#     netinfo: Netinfo,
#     image_path: str,
#     stage: str,
#     run_sim: bool,
# ) :
#     """Load and resize the input image to match the model's first input tensor.

#     Returns the array reshaped to the network specification along with the original BGR image.
#     """
#     raw_image = cv2.imread(image_path)
#     if raw_image is None:
#         raise FileNotFoundError(f"Unable to load image: {image_path}")

#     if not netinfo.i_cubic:
#         raise RuntimeError("The loaded network does not expose any image-like input tensors.")

#     target_shape = netinfo.i_shape[0]
#     cubic = netinfo.i_cubic[0]
#     print(cubic.w,cubic.h,cubic.c)
#     resized = cv2.resize(raw_image, (cubic.w, cubic.h))

#     # Keep the channel dimension consistent with the network definition.
#     if resized.ndim == 2:
#         resized = resized[:, :, None]
#     if resized.shape[2] != cubic.c:
#         resized = resized[:, :, : cubic.c]

#     data = resized
#     if stage not in ("a", "g") and run_sim:
#         data = resized.astype(np.float32)

#     return data.reshape(target_shape), 

def _prepare_input_image(
    netinfo: Netinfo,
    image_path: str,
    stage: str,
    run_sim: bool,
):
    """Load image and apply YOLO-style letterbox to match model input size."""
    raw_image = cv2.imread(image_path)
    if raw_image is None:
        raise FileNotFoundError(f"Unable to load image: {image_path}")

    if not netinfo.i_cubic:
        raise RuntimeError("The loaded network does not expose any image-like input tensors.")

    cubic = netinfo.i_cubic[0]
    target_w, target_h, target_c = cubic.w, cubic.h, cubic.c
    h0, w0 = raw_image.shape[:2]

    # ---- Letterbox resize ----
    scale = min(target_w / w0, target_h / h0)
    new_w, new_h = int(round(w0 * scale)), int(round(h0 * scale))
    resized = cv2.resize(raw_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 创建灰色画布
    canvas = np.full((target_h, target_w, 3), 128, dtype=resized.dtype)
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
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    canvas = image.copy()

    for det in detections:
        cls_idx, score, x1, y1, x2, y2 = det
        x1_i = int(round(x1))
        y1_i = int(round(y1))
        x2_i = int(round(x2))
        y2_i = int(round(y2))

        x1_i = max(0, min(canvas.shape[1] - 1, x1_i))
        y1_i = max(0, min(canvas.shape[0] - 1, y1_i))
        x2_i = max(0, min(canvas.shape[1] - 1, x2_i))
        y2_i = max(0, min(canvas.shape[0] - 1, y2_i))

        if x2_i < x1_i:
            x1_i, x2_i = x2_i, x1_i
        if y2_i < y1_i:
            y1_i, y2_i = y2_i, y1_i

        cv2.rectangle(canvas, (x1_i, y1_i), (x2_i, y2_i), (0, 255, 0), 2)
        label = f"{cls_idx}:{score:.2f}"
        cv2.putText(
            canvas,
            label,
            (x1_i, max(0, y1_i - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    if not cv2.imwrite(save_path, canvas):
        raise IOError(f"Failed to write visualization image to {save_path}")
    mprint(f"Saved visualization to {save_path}", VERBOSE, 0)


def main() -> None:
    args = _parse_args()

    json_path, raw_path = getJrPath(args.model_dir, args.stage, args.sim)
    mprint(f"Using model files: {json_path} / {raw_path}", VERBOSE, 0)

    network = loadNetwork(json_path, raw_path)
    netinfo = Netinfo(network)
    network_view = network.view(netinfo.inp_shape_opid + 1)

    mmu_required = netinfo.mmu or args.enable_mmu
    custom_ops = fpgaOPlist(network)
    has_imk = "customop::ImageMakeNode" in custom_ops
    mprint(f"Detected custom ops: {sorted(custom_ops)}", VERBOSE, 1)
    mprint(f"ImageMakeNode enabled: {has_imk}", VERBOSE, 0)

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

    # input_data, raw_image = _prepare_input_image(netinfo, args.image, args.stage, args.sim)
    # original_hw = raw_image.shape[:2]
    input_data, raw_image, meta = _prepare_input_image(netinfo, args.image, args.stage, args.sim)

    input_tensor = numpy2Tensor(input_data, network)

    if not args.sim:
        hardware_dma_init(device, input_tensor, netinfo.i_shape[0][1:], has_imk)

    outputs = session.forward([input_tensor])

    if args.hard_layout and not args.sim:
        try:
            device.reset(0)
        except AttributeError:
            mprint("Device reset not supported on this backend.", VERBOSE, 0)

    output_arrays: List[np.ndarray] = []
    for index, tensor in enumerate(outputs):
        try:
            array = np.asarray(tensor.to(HostDevice.MemRegion()))
        except Exception:  # pragma: no cover - conversion fallback
            array = np.asarray(tensor)
        output_arrays.append(array)
        mprint(
            f"Output {index}: shape={array.shape}, dtype={array.dtype}",
            VERBOSE,
            0,
        )

    # detections = _decode_detections(
    #     output_arrays,
    #     netinfo,
    #     args.conf,
    #     args.iou,
    #     args.num_classes,
    #     args.multi_label,
    #     original_hw,
    # )
    if args.hard_layout:
        detections = _decode_detections_hard(
            output_arrays,
            network,
            meta,
            args.conf,
            args.iou,
            args.num_classes,
        )
    else:
        detections = _decode_detections(
            output_arrays,
            netinfo,
            args.conf,
            args.iou,
            args.num_classes,
            args.multi_label,
            meta,
        )
    
    if detections:
        mprint("Detections [cls, conf, x1, y1, x2, y2]", VERBOSE, 0)
        for det in detections:
            mprint(str(det), VERBOSE, 0)
    else:
        mprint("No detections above the confidence threshold.", VERBOSE, 0)

    print("Detections:", detections)
    output_path = args.save_vis
    if not output_path:
        base, ext = os.path.splitext(args.image)
        output_path = f"{base}_det{ext or '.jpg'}"
    _visualize_detections(raw_image, detections, output_path)

    if not args.sim:
        Device.Close(device)
    
    print("Demo inference completed successfully.")


if __name__ == "__main__":
    main()
