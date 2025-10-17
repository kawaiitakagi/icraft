#!/usr/bin/env python
import argparse
import torch
import os
from gaze_estimation import create_model, get_default_config

RED = '\033[31m'       # 设置前景色为红色
RESET = '\033[0m'      # 重置所有属性到默认值
ver = torch.__version__
assert ("2.0.1" in ver) or ("1.9" in ver), f"{RED}Unsupported PyTorch version: {ver}{RESET}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config/demo_mpiigaze_resnet.yaml",type=str)
    parser.add_argument('--weight', default="../weights/mpiigaze_resnet_preact.pth",type=str)
    parser.add_argument('--output-path', '-o', default="../2_compile/fmodel/", type=str)
    parser.add_argument('--trace_model', default="MPIIGaze_1x1x36x60_1x2.onnx", type=str)
    args = parser.parse_args()
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    config = get_default_config()
    config.merge_from_file(args.config)

    device = torch.device(config.device)

    model = create_model(config)
    if args.weight is not None:
        checkpoint = torch.load(args.weight, map_location=device)
        model.load_state_dict(checkpoint['model'])
    model.eval()

    if config.mode == 'MPIIGaze':
        x = torch.zeros((1, 1, 36, 60), dtype=torch.float32, device=device)
        y = torch.zeros((1, 2), dtype=torch.float32, device=device)
        data = (x, y)
        
    elif config.mode == 'MPIIFaceGaze':
        x = torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=device)
        data = (x, )
    else:
        raise ValueError

    trace_model = args.output_path + args.trace_model
    torch.onnx.export(model, data, trace_model, opset_version=11)
    print("successful export model to ", trace_model)


if __name__ == '__main__':
    main()
