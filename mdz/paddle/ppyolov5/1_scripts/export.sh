#!/bin/sh

# 导出模型运行

python 1_save.py

paddle2onnx --model_dir output_inference/yolov5_s_300e_coco --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 11 --save_file ../2_compile/fmodel/ppyolov5s-640x640.onnx