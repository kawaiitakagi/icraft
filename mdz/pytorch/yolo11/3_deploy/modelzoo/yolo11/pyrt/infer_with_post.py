"""YOLO11 inference entry point that wires up the detpost hardware decoder.

This script mirrors the behaviour of the C++ deployment tool by consuming the
same YAML configuration files and invoking the Python port of
``post_detpost_hard``. It uses the runtime helpers from ``pyrtutils`` to load a
compiled model, run inference either on hardware or in simulation, and perform
post-processing on the hardware-formatted detpost outputs.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import yaml

# Allow importing the shared runtime helpers shipped under ``Deps/modelzoo``.
CURRENT_DIR = Path(__file__).resolve().parent
DEPS_DIR = CURRENT_DIR.parents[1] / "Deps" / "modelzoo"
if str(DEPS_DIR) not in sys.path:
    sys.path.insert(0, str(DEPS_DIR))

from pyrtutils.icraft_utils import (  # type: ignore  # pylint: disable=wrong-import-position
    getJrPath,
    initSession,
    loadNetwork,
    numpy2Tensor,
    openDevice,
)
from pyrtutils.Netinfo import Netinfo  # type: ignore  # pylint: disable=wrong-import-position
from pyrtutils.utils import (  # type: ignore  # pylint: disable=wrong-import-position
    VERBOSE,
    dmaInit,
    mprint,
)
from pyrtutils.utils.io import checkDir  # type: ignore  # pylint: disable=wrong-import-position
from icraft.xrt import Device  # type: ignore  # pylint: disable=wrong-import-position

from .postprocess_yolo11 import get_stride, post_detpost_hard


class PicPre:
    """Light-weight Python reimplementation of the C++ ``PicPre`` helper."""

    BOTH_SIDE = 0
    LONG_SIDE = 1
    SHORT_SIDE = 2

    BR = 0
    AROUND = 1

    def __init__(self, image: np.ndarray) -> None:
        if image is None:
            raise ValueError("Input image cannot be None")
        self.ori_img = image.copy()
        if image.ndim == 3 and image.shape[2] == 3:
            self.src_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            self.src_img = image.copy()
        self.dst_img = self.src_img.copy()
        self._dst_shape = (self.src_img.shape[0], self.src_img.shape[1])
        self._real_resized_hw = (self.src_img.shape[0], self.src_img.shape[1])
        self._real_resized_ratio = (1.0, 1.0)
        self._pad_info = (0, 0)

    def Resize(
        self,
        dst_shape_hw: Tuple[int, int],
        mode: int = LONG_SIDE,
        interpolation: int = cv2.INTER_LINEAR,
    ) -> "PicPre":
        target_h, target_w = int(dst_shape_hw[0]), int(dst_shape_hw[1])
        if target_h <= 0 or target_w <= 0:
            raise ValueError("Destination shape must be positive")

        ori_h, ori_w = self.src_img.shape[:2]
        ratio_h = target_h / float(ori_h)
        ratio_w = target_w / float(ori_w)

        if mode == self.BOTH_SIDE:
            resized_h, resized_w = target_h, target_w
            ratio_h_final, ratio_w_final = ratio_h, ratio_w
        elif mode == self.LONG_SIDE:
            ratio = min(ratio_h, ratio_w)
            resized_h = int(round(ori_h * ratio))
            resized_w = int(round(ori_w * ratio))
            ratio_h_final = ratio_w_final = ratio
        elif mode == self.SHORT_SIDE:
            ratio = max(ratio_h, ratio_w)
            resized_h = int(round(ori_h * ratio))
            resized_w = int(round(ori_w * ratio))
            ratio_h_final = ratio_w_final = ratio
        else:
            raise ValueError(f"Unsupported resize mode: {mode}")

        resized_h = max(resized_h, 1)
        resized_w = max(resized_w, 1)

        self.dst_img = cv2.resize(self.src_img, (resized_w, resized_h), interpolation)
        self._dst_shape = (target_h, target_w)
        self._real_resized_hw = (resized_h, resized_w)
        self._real_resized_ratio = (ratio_h_final, ratio_w_final)
        return self

    def rPad(self, pad_mode: int = BR, value: int = 114) -> "PicPre":
        target_h, target_w = self._dst_shape
        resized_h, resized_w = self._real_resized_hw
        dh = max(target_h - resized_h, 0)
        dw = max(target_w - resized_w, 0)

        if pad_mode == self.BR:
            top = 0
            left = 0
            bottom = int(dh)
            right = int(dw)
        elif pad_mode == self.AROUND:
            top = int(math.floor(dh / 2.0))
            bottom = int(dh - top)
            left = int(math.floor(dw / 2.0))
            right = int(dw - left)
        else:
            raise ValueError(f"Unsupported pad mode: {pad_mode}")

        if self.dst_img.ndim == 2 or self.dst_img.shape[2] == 1:
            border_value = int(value)
        else:
            border_value = (int(value), int(value), int(value))

        self.dst_img = cv2.copyMakeBorder(
            self.dst_img,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=border_value,
        )
        self._pad_info = (left, top)
        return self

    def getRatio(self) -> Tuple[float, float]:
        return self._real_resized_ratio

    def getPad(self) -> Tuple[int, int]:
        return self._pad_info


def _resolve_path(base: Path, path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str((base / candidate).resolve())


def _load_label_names(path: str) -> List[str]:
    if not path:
        return []
    names_path = Path(path)
    if not names_path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")
    with names_path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _load_image_list(path: str) -> List[str]:
    list_path = Path(path)
    if not list_path.exists():
        raise FileNotFoundError(f"Image list not found: {path}")
    with list_path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _prepare_tensor_data(pic: PicPre, netinfo: Netinfo, stage: str, run_sim: bool) -> np.ndarray:
    array = pic.dst_img
    if stage not in ("a", "g") and run_sim:
        array = array.astype(np.float32, copy=False)
    shape = netinfo.i_shape[0]
    if len(shape) != array.ndim + 1:
        raise ValueError(f"Input tensor rank mismatch: expected {shape}, got {array.shape}")
    # The runtime expects NHWC layout as reflected by Netinfo.i_cubic
    array = array.reshape(shape)
    return array


def _set_norm_by_head(num_heads: int, parts: int, normalratio: Sequence[float]) -> List[List[float]]:
    flat = list(normalratio)
    if len(flat) != num_heads * parts:
        raise ValueError(
            f"Norm list length mismatch: expected {num_heads * parts}, got {len(flat)}"
        )
    grouped: List[List[float]] = []
    for head in range(num_heads):
        start = head * parts
        grouped.append(flat[start : start + parts])
    return grouped


def _get_real_out_channels(
    ori_out_channels: Sequence[int], bits: int, num_classes: int
) -> List[int]:
    if bits == 8:
        maxc, minc = 64, 8
    elif bits == 16:
        maxc, minc = 32, 4
    else:
        raise ValueError(f"Unsupported detpost bit width: {bits}")

    def last_c(ori_c: float) -> int:
        return int(math.ceil(float(ori_c) / float(minc)) * minc + minc)

    def mid_c(ori_c: float) -> int:
        return int(math.ceil(float(ori_c) / float(maxc)) * maxc)

    parts = len(ori_out_channels)
    if parts == 1:
        one_anchor = float(ori_out_channels[0]) / float(num_classes)
        anchor_length = last_c(one_anchor)
        return [int(num_classes * anchor_length)]
    if parts == 2:
        return [mid_c(ori_out_channels[0]), last_c(ori_out_channels[1])]
    if parts == 3:
        return [
            mid_c(ori_out_channels[0]),
            mid_c(ori_out_channels[1]),
            last_c(ori_out_channels[2]),
        ]
    raise ValueError("DetPost supports up to three channel partitions")


def _parse_args() -> argparse.Namespace:
    default_cfg = CURRENT_DIR.parent / "cfg" / "yolo11s.yaml"
    parser = argparse.ArgumentParser(description="Run YOLO11 inference with detpost post-processing.")
    parser.add_argument(
        "config",
        nargs="?",
        default=str(default_cfg),
        help="Path to the YAML configuration file (defaults to yolo11s.yaml).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg is None:
        raise ValueError("Configuration file is empty")

    cfg_dir = config_path.parent

    imodel_cfg = cfg.get("imodel", {})
    dataset_cfg = cfg.get("dataset", {})
    param_cfg = cfg.get("param", {})

    run_sim = bool(imodel_cfg.get("sim", False))
    stage = str(imodel_cfg.get("stage", "g"))

    model_dir = _resolve_path(cfg_dir, imodel_cfg.get("dir", "."))
    json_path, raw_path = getJrPath(model_dir, stage, run_sim)
    mprint(f"Using model files: {json_path} / {raw_path}", VERBOSE, 0)

    network = loadNetwork(json_path, raw_path)
    netinfo = Netinfo(network)
    network_view = network.view(netinfo.inp_shape_opid + 1)

    mmu_required = bool(imodel_cfg.get("mmu", False)) or netinfo.mmu
    device = openDevice(
        run_sim,
        str(imodel_cfg.get("ip", "0.0.0.0")),
        mmu_Mode=mmu_required,
        cuda_Mode=bool(imodel_cfg.get("cudamode", False)),
        npu_addr=str(imodel_cfg.get("npu_addr", "0x40000000")),
        dma_addr=str(imodel_cfg.get("dma_addr", "0x80000000")),
    )

    session = initSession(
        run_sim,
        network_view,
        device,
        mmu_required,
        imodel_cfg.get("speedmode", False),
        imodel_cfg.get("compressFtmp", False),
    )
    session.enableTimeProfile(True)
    session.apply()

    img_root = _resolve_path(cfg_dir, dataset_cfg.get("dir", ""))
    img_list_path = _resolve_path(cfg_dir, dataset_cfg.get("list", ""))
    names_path = _resolve_path(cfg_dir, dataset_cfg.get("names", ""))
    res_root = _resolve_path(cfg_dir, dataset_cfg.get("res", "output"))

    checkDir(res_root)
    labels = _load_label_names(names_path)
    image_names = _load_image_list(img_list_path)

    conf = float(param_cfg.get("conf", 0.25))
    iou_thresh = float(param_cfg.get("iou_thresh", 0.7))
    multi_label = bool(param_cfg.get("multilabel", False))
    num_classes = int(param_cfg.get("number_of_class", 80))
    num_heads = int(param_cfg.get("number_of_head", len(netinfo.o_cubic)))
    anchors = param_cfg.get("anchors", []) or []
    fpga_nms = bool(param_cfg.get("fpga_nms", False))
    bbox_info_channel = int(param_cfg.get("bbox_info_channel", 64))

    if not netinfo.DetPost_on:
        raise RuntimeError("The loaded network does not enable DetPost; hardware layout decoding is unavailable.")

    normalratio = netinfo.o_scale
    if not normalratio:
        raise RuntimeError("Output norm ratios are missing from the network metadata.")

    ori_out_channels = [num_classes, bbox_info_channel]
    parts = len(ori_out_channels)
    norm = _set_norm_by_head(num_heads, parts, normalratio)
    real_out_channels = _get_real_out_channels(ori_out_channels, netinfo.detpost_bit, num_classes)
    stride_list = get_stride(netinfo)

    show = bool(imodel_cfg.get("show", False))
    save = bool(imodel_cfg.get("save", True))

    cubic = netinfo.i_cubic[0]
    target_shape = (cubic.h, cubic.w)

    for idx, name in enumerate(image_names, start=1):
        image_path = Path(img_root) / name
        raw_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if raw_image is None:
            mprint(f"Warning: failed to read image {image_path}", VERBOSE, 0)
            continue

        pic = PicPre(raw_image)
        pic.Resize(target_shape, PicPre.LONG_SIDE).rPad(PicPre.BR)

        tensor_data = _prepare_tensor_data(pic, netinfo, stage, run_sim)
        input_tensor = numpy2Tensor(tensor_data, network)

        if hasattr(device, "getMemRegion") and netinfo.ImageMake_on:
            dmaInit(device, input_tensor, netinfo.i_shape[0][1:], netinfo.ImageMake_on)

        outputs = session.forward([input_tensor])

        if not run_sim and hasattr(device, "reset"):
            try:
                device.reset(1)
            except Exception:
                pass

        post_detpost_hard(
            outputs,
            pic,
            netinfo,
            conf,
            iou_thresh,
            multi_label,
            fpga_nms,
            num_classes,
            anchors,
            labels,
            show,
            save,
            res_root,
            name,
            device,
            run_sim,
            norm,
            real_out_channels,
            stride_list,
            bbox_info_channel,
        )

        mprint(f"Processed {idx}/{len(image_names)}: {name}", VERBOSE, 0)

    if not run_sim:
        Device.Close(device)


if __name__ == "__main__":
    main()
