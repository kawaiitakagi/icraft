# Copyright (c) OpenMMLab. All rights reserved.
import sys
sys.path.append(R"../0_MMPSPNet")
from argparse import ArgumentParser
from mmengine.model import revert_sync_batchnorm
from mmseg.apis import init_model,inference_model, show_result_pyplot
import torch
from mmengine.dataset import Compose
from mmseg.models.decode_heads.psp_head import PSPHead

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img',default="../0_MMPSPNet/demo/demo.png", help='Image file')
    parser.add_argument('--config',default="../0_MMPSPNet/configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py", help='Config file')
    parser.add_argument('--checkpoint',default="../weights/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth", help='Checkpoint file')
    parser.add_argument('--out-file', default="./outputs/result_infer.png", help='Path to output file')
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

def new_forward(self, inputs):
    """Forward function."""
    output = self._forward_feature(inputs)
    output = self.cls_seg(output)
    return RES[0]

def pre_processor(args, model):
    cfg = model.cfg
    for t in cfg.test_pipeline:
        if t.get('type') == 'LoadAnnotations':
            cfg.test_pipeline.remove(t)
    pipeline = Compose(cfg.test_pipeline)
    # prepare data
    data_ = dict(img_path=args.img)
    data = pipeline(data_)
    inputs = data["inputs"]
    inputs = inputs[[2, 1, 0], ...].float()## BGR -> RGB
    # # 将tensor转换为NumPy数组
    # image_np = torchvision.transforms.functional.to_pil_image(inputs)
    # # 将NumPy数组转换为BGR格式
    # image_bgr = cv2.cvtColor(np.array(image_np), cv2.COLOR_RGB2BGR)
    # # 保存图像
    # cv2.imwrite('output.jpg', image_bgr)

    mean = cfg._cfg_dict["data_preprocessor"]["mean"]
    std = cfg._cfg_dict["data_preprocessor"]["std"]
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    data_out = (inputs - mean) / std
    return data_out

RES = []

if __name__ == '__main__':
    args = parse_args()

    # 加载模型
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    # 前处理
    data = pre_processor(args, model)
    input = data.unsqueeze(0)

    # 加载trace model
    model_traced =torch.jit.load("../2_compile/fmodel/mmpspnet_traced_1024x2048.pt", map_location="cpu")

    # 前向推理
    res = model_traced(input)
    RES.append(res)

    PSPHead.forward = new_forward

    result = inference_model(model, args.img)

    show_result_pyplot(
        model,
        args.img,
        result,
        title=args.title,
        opacity=args.opacity,
        draw_gt=False,
        show=False if args.out_file is not None else True,
        out_file=args.out_file)


