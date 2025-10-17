# Copyright (c) OpenMMLab. All rights reserved.
import sys
sys.path.append(R"../0_MMPSPNet")
from argparse import ArgumentParser
from mmengine.model import revert_sync_batchnorm
from mmseg.apis import inference_model, init_model, show_result_pyplot

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img',default="../0_MMPSPNet/demo/demo.png", help='Image file')
    parser.add_argument('--config',default="../0_MMPSPNet/configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py", help='Config file')
    parser.add_argument('--checkpoint',default="../weights/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth", help='Checkpoint file')
    parser.add_argument('--out-file', default="./outputs/result.png", help='Path to output file')
    parser.add_argument(
        '--device', default='cpu', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--title', default='result', help='The image identifier.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)
    # test a single image
    result = inference_model(model, args.img)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result,
        title=args.title,
        opacity=args.opacity,
        draw_gt=False,
        show=False if args.out_file is not None else True,
        out_file=args.out_file)
