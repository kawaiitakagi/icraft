# Copyright (c) OpenMMLab. All rights reserved.
import sys
sys.path.append(R"../0_MMPSPNet")
from argparse import ArgumentParser
from mmengine.model import revert_sync_batchnorm
from mmseg.apis import init_model
import torch
from mmseg.models.decode_heads.psp_head import PPM
from mmseg.models.utils import resize

RED = '\033[31m'       # 设置前景色为红色
RESET = '\033[0m'      # 重置所有属性到默认值
ver = torch.__version__
assert ("1.6" in ver) or ("1.9" in ver), f"{RED}Unsupported PyTorch version: {ver}{RESET}"

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img',default="../0_MMPSPNet/demo/demo.png", help='Image file')
    parser.add_argument('--config',default="../0_MMPSPNet/configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py", help='Config file')
    parser.add_argument('--checkpoint',default="../weights/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth", help='Checkpoint file')
    parser.add_argument('--out-file', default="./outputs/", help='Path to output file')
    parser.add_argument("--trace_path", type=str, default="../2_compile/fmodel/mmpspnet_traced_1024x2048.pt", help="path of traced model")
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

def new_forward(self, x):
    """Psp_head.py new Forward function."""
    ppm_outs = []
    for ppm in self:
        ppm_out = ppm(x)#[1,512,1,1]
        size_final = x.size()[2:]
        upsampled_ppm_out = resize(
            ppm_out,
            size=(16,32),
            mode='bilinear',
            align_corners=self.align_corners)##False
        upsampled_ppm_out = resize(
            upsampled_ppm_out,
            size=(64,128),
            mode='bilinear',
            align_corners=self.align_corners)##False
        upsampled_ppm_out = resize(
            upsampled_ppm_out,
            size=size_final,##[128,256]
            mode='bilinear',
            align_corners=self.align_corners)##False
        ppm_outs.append(upsampled_ppm_out)
    return ppm_outs
   
if __name__ == '__main__':
    args = parse_args()

    # 加载模型
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    input = torch.randn(1,3,1024,2048)

    # 等效替换upsample
    PPM.__call__ = new_forward
    
    out = model(input) 
    print(out.shape)##没有接后处理,最后一个conv:19*512->out:([1, 19, 128,256])
    
    # 删除前处理
    del model.data_preprocessor
    del model.auxiliary_head

    # 将self.dropout置为None, 前向过程中不再调用self.dropout
    model.decode_head.dropout = None 

    # 模型导出
    tmodel = torch.jit.trace(model,input)
    torch.jit.save(tmodel, args.trace_path)
    print("successful save model in ", args.trace_path)
    

