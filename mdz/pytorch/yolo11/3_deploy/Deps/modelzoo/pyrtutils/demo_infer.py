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

from icraft.host_backend import HostDevice  # noqa: E402  pylint: disable=wrong-import-position
from icraft.xrt import Device  # noqa: E402  pylint: disable=wrong-import-position


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64, copy=False)
    return 1.0 / (1.0 + np.exp(-x64))


def _dfl(dist: np.ndarray, value_range: np.ndarray) -> np.ndarray:
    reshaped = dist.reshape(4, -1).astype(np.float64, copy=False)
    logits = reshaped - reshaped.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return (probs * value_range).sum(axis=1)


def _decode_detections(
    outputs: List[np.ndarray],
    netinfo: Netinfo,
    conf_thresh: float,
    iou_thresh: float,
    num_classes: Optional[int],
    multi_label: bool,
    original_hw: Tuple[int, int],
) -> List[List[float]]:
    if len(outputs) % 2 != 0:
        raise RuntimeError(
            "The demo expects detection models that produce class and box tensors in pairs."
        )

    input_h = netinfo.i_cubic[0].h
    input_w = netinfo.i_cubic[0].w
    scale_y = original_hw[0] / float(input_h)
    scale_x = original_hw[1] / float(input_w)

    all_boxes: List[List[float]] = []
    all_scores: List[float] = []
    all_ids: List[int] = []

    for head in range(0, len(outputs), 2):
        cls_pred = np.asarray(outputs[head]).squeeze()
        box_pred = np.asarray(outputs[head + 1]).squeeze()

        if cls_pred.ndim != 3 or box_pred.ndim != 3:
            raise RuntimeError(
                "Unexpected tensor layout. Expected [H, W, C] for each detection head."
            )

        height, width, channels = cls_pred.shape
        head_num_classes = channels if num_classes is None else num_classes

        stride_y = input_h / float(height)
        stride_x = input_w / float(width)

        reg_max = box_pred.shape[-1] // 4
        value_range = np.arange(reg_max, dtype=np.float64)

        for y in range(height):
            for x in range(width):
                class_logits = cls_pred[y, x, :head_num_classes]
                class_scores = _sigmoid(class_logits)

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
        all_boxes,
        all_scores,
        all_ids,
        conf=conf_thresh,
        iou=iou_thresh,
        NOC=classes,
    )

    detections = []
    for cls_idx, score, box in zip(nms_ids, nms_scores, nms_boxes):
        x1 = box[0] * scale_x
        y1 = box[1] * scale_y
        x2 = box[2] * scale_x
        y2 = box[3] * scale_y
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
        required=True,
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
        default=0.25,
        help="Confidence threshold applied before Soft-NMS.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
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
    return parser.parse_args()


def _prepare_input_image(
    netinfo: Netinfo,
    image_path: str,
    stage: str,
    run_sim: bool,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Load and resize the input image to match the model's first input tensor."""
    raw_image = cv2.imread(image_path)
    if raw_image is None:
        raise FileNotFoundError(f"Unable to load image: {image_path}")

    if not netinfo.i_cubic:
        raise RuntimeError("The loaded network does not expose any image-like input tensors.")

    target_shape = netinfo.i_shape[0]
    cubic = netinfo.i_cubic[0]
    resized = cv2.resize(raw_image, (cubic.w, cubic.h))

    # Keep the channel dimension consistent with the network definition.
    if resized.ndim == 2:
        resized = resized[:, :, None]
    if resized.shape[2] != cubic.c:
        resized = resized[:, :, : cubic.c]

    data = resized
    if stage not in ("a", "g") and run_sim:
        data = resized.astype(np.float32)

    return data.reshape(target_shape), raw_image.shape[:2]


def main() -> None:
    args = _parse_args()

    json_path, raw_path = getJrPath(args.model_dir, args.stage, args.sim)
    mprint(f"Using model files: {json_path} / {raw_path}", VERBOSE, 0)

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

    input_data, original_hw = _prepare_input_image(netinfo, args.image, args.stage, args.sim)
    input_tensor = numpy2Tensor(input_data, network)

    if not args.sim:
        dmaInit(args.sim, netinfo.ImageMake_on, netinfo.i_shape[0][1:], input_tensor, device)

    outputs = session.forward([input_tensor])

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

    detections = _decode_detections(
        output_arrays,
        netinfo,
        args.conf,
        args.iou,
        args.num_classes,
        args.multi_label,
        original_hw,
    )

    if detections:
        mprint("Detections [cls, conf, x1, y1, x2, y2]", VERBOSE, 0)
        for det in detections:
            mprint(str(det), VERBOSE, 0)
    else:
        mprint("No detections above the confidence threshold.", VERBOSE, 0)

    if not args.sim:
        Device.Close(device)


if __name__ == "__main__":
    main()
