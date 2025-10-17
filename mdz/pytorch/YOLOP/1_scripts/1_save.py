import torch
import torch.nn as nn
from lib.models.common import Conv, SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect, SharpenConv
from torch.nn import Upsample
from lib.utils import check_anchor_order
from lib.utils import initialize_weights
import argparse
import math
import torch.nn.functional as F

RED = '\033[31m'       # 设置前景色为红色
RESET = '\033[0m'      # 重置所有属性到默认值
ver = torch.__version__
assert ("1.9" in ver) or ("2.0.1" in ver), f"{RED}Unsupported PyTorch version: {ver}{RESET}"

# The lane line and the driving area segment branches without share information with each other and without link
YOLOP = [
    [24, 33, 42],  # Det_out_idx, Da_Segout_idx, LL_Segout_idx
    [-1, Focus, [3, 32, 3]],  # 0
    [-1, Conv, [32, 64, 3, 2]],  # 1
    [-1, BottleneckCSP, [64, 64, 1]],  # 2
    [-1, Conv, [64, 128, 3, 2]],  # 3
    [-1, BottleneckCSP, [128, 128, 3]],  # 4
    [-1, Conv, [128, 256, 3, 2]],  # 5
    [-1, BottleneckCSP, [256, 256, 3]],  # 6
    [-1, Conv, [256, 512, 3, 2]],  # 7
    [-1, SPP, [512, 512, [5, 9, 13]]],  # 8 SPP
    [-1, BottleneckCSP, [512, 512, 1, False]],  # 9
    [-1, Conv, [512, 256, 1, 1]],  # 10
    [-1, Upsample, [None, 2, 'nearest']],  # 11
    [[-1, 6], Concat, [1]],  # 12
    [-1, BottleneckCSP, [512, 256, 1, False]],  # 13
    [-1, Conv, [256, 128, 1, 1]],  # 14
    [-1, Upsample, [None, 2, 'nearest']],  # 15
    [[-1, 4], Concat, [1]],  # 16         #Encoder

    [-1, BottleneckCSP, [256, 128, 1, False]],  # 17
    [-1, Conv, [128, 128, 3, 2]],  # 18
    [[-1, 14], Concat, [1]],  # 19
    [-1, BottleneckCSP, [256, 256, 1, False]],  # 20
    [-1, Conv, [256, 256, 3, 2]],  # 21
    [[-1, 10], Concat, [1]],  # 22
    [-1, BottleneckCSP, [512, 512, 1, False]],  # 23
    [[17, 20, 23], Detect,
     [1, [[3, 9, 5, 11, 4, 20], [7, 18, 6, 39, 12, 31], [19, 50, 38, 81, 68, 157]], [128, 256, 512]]],
    # Detection head 24: from_(features from specific layers), block, nc(num_classes) anchors ch(channels)

    [16, Conv, [256, 128, 3, 1]],  # 25
    [-1, Upsample, [None, 2, 'nearest']],  # 26
    [-1, BottleneckCSP, [128, 64, 1, False]],  # 27
    [-1, Conv, [64, 32, 3, 1]],  # 28
    [-1, Upsample, [None, 2, 'nearest']],  # 29
    [-1, Conv, [32, 16, 3, 1]],  # 30
    [-1, BottleneckCSP, [16, 8, 1, False]],  # 31
    [-1, Upsample, [None, 2, 'nearest']],  # 32
    [-1, Conv, [8, 2, 3, 1]],  # 33 Driving area segmentation head

    [16, Conv, [256, 128, 3, 1]],  # 34
    [-1, Upsample, [None, 2, 'nearest']],  # 35
    [-1, BottleneckCSP, [128, 64, 1, False]],  # 36
    [-1, Conv, [64, 32, 3, 1]],  # 37
    [-1, Upsample, [None, 2, 'nearest']],  # 38
    [-1, Conv, [32, 16, 3, 1]],  # 39
    [-1, BottleneckCSP, [16, 8, 1, False]],  # 40
    [-1, Upsample, [None, 2, 'nearest']],  # 41
    [-1, Conv, [8, 2, 3, 1]]  # 42 Lane line segmentation head
]


class MCnet(nn.Module):
    def __init__(self, block_cfg):
        super(MCnet, self).__init__()
        layers, save = [], []
        self.nc = 1  # traffic or not
        self.detector_index = -1
        self.det_out_idx = block_cfg[0][0]
        self.seg_out_idx = block_cfg[0][1:]
        self.num_anchors = 3
        self.num_outchannel = 5 + self.nc  # dx,dy,dw,dh,obj_conf+cls_conf
        # Build model
        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            block = eval(block) if isinstance(block, str) else block  # eval strings
            if block is Detect:
                self.detector_index = i
            block_ = block(*args)
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist
        assert self.detector_index == block_cfg[0][0]

        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self.names = [str(i) for i in range(self.nc)]

        # set stride、anchor for detector
        Detector = self.model[self.detector_index]  # detector
        if isinstance(Detector, Detect):
            s = 128  # 2x min stride
            # for x in self.forward(torch.zeros(1, 3, s, s)):
            #     print (x.shape)
            with torch.no_grad():
                model_out = self.forward(torch.zeros(1, 3, s, s))
                detects, _, _ = model_out
                Detector.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
            # print("stride"+str(Detector.stride ))
            Detector.anchors /= Detector.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale
            check_anchor_order(Detector)
            self.stride = Detector.stride
            # self._initialize_biases()
        initialize_weights(self)

    def forward(self, x):
        cache = []
        out = []
        det_out = None
        for i, block in enumerate(self.model):
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) \
                    else [x if j == -1 else cache[j] for j in
                          block.from_]  # calculate concat detect
            x = block(x)
            if i in self.seg_out_idx:  # save driving area segment result
                # m = nn.Sigmoid()
                # out.append(m(x))
                out.append(torch.sigmoid(x))
            if i == self.detector_index:
                det_out = x
            cache.append(x if block.index in self.save else None)
        return det_out, out[0], out[1]  # det, da, ll
        # (1,na*ny*nx*nl,no=2+2+1+nc=xy+wh+obj_conf+cls_prob), (1,2,h,w) (1,2,h,w)

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.model[self.detector_index]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


# Icraft适配，需完成以下模型修改：
#----------------------------
# 修改点1：等效Focus
#----------------------------
FOCUS_WEIGHTS=torch.zeros(12,3,2,2)
for i in range(12):
    j=i%3
    k=(i//3)%2
    l=0 if i<6 else 1
    FOCUS_WEIGHTS[i,j,k,l]=1
FOCUS_BIAS = torch.zeros(12)         
def FOCUS_forward(self,x):
    y = F.conv2d(x,FOCUS_WEIGHTS,bias=FOCUS_BIAS,stride=2,padding=0,dilation=1,groups=1)
    y = self.conv(y)
    return y
Focus.forward = FOCUS_forward

#----------------------------
# 修改点2：detect头只保留到卷积
#----------------------------
def Detect_forward(self, x):
    for i in range(self.nl):
        x[i] = self.m[i](x[i])  # conv
        # print(str(i)+str(x[i].shape))
    return x 
Detect.forward = Detect_forward

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--height', type=int, default=384)  # height
    parser.add_argument('--width', type=int, default=640)  # width
    args = parser.parse_args()

    do_simplify = True

    device = 'cpu'
    model = MCnet(YOLOP)
    checkpoint = torch.load('./weights/End-to-end.pth', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    height = args.height
    width = args.width
    print("Load ./weights/End-to-end.pth done!")
    onnx_path = f'../2_compile/fmodel/yolop_1x3x{height}x{width}_traced.onnx'
    inputs = torch.randn(1, 3, height, width)

    print(f"Converting to {onnx_path}")
    torch.onnx.export(model, inputs, onnx_path,
                      verbose=False, opset_version=11, input_names=['images'],
                      output_names=['det_out1', 'det_out2','det_out3', 'drive_area_seg', 'lane_line_seg'])
    print('convert', onnx_path, 'to onnx finish!!!')


