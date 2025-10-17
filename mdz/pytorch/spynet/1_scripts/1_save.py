#!/usr/bin/env python
import sys
sys.path.append(R"../0_spynet")
import getopt
import math
import numpy
import PIL
import PIL.Image
import sys
import torch
from flo2img import *
##########################################################

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = False # make sure to use cudnn for computational performance
INP_H = 320 #544 352 320 480 448 
INP_W = 544 #960 640 576 864 832
TRACED_MODEL_PATH = "../2_compile/fmodel/spynet_"+str(INP_H)+"x"+str(INP_W)+".pt"
##########################################################
WEIGHT_PATH = "../weights/network-sintel-final.pytorch"
args_strOne = '../0_spynet/images/one.png'
# args_strOne = './images/one.png'
args_strTwo = '../0_spynet/images/two.png'
# args_strTwo = './images/two.png'

for strOption, strArg in getopt.getopt(sys.argv[1:], '', [
    'one=',
    'two=',
])[0]:
    if strOption == '--one' and strArg != '': args_strOne = strArg # path to the first frame
    if strOption == '--two' and strArg != '': args_strTwo = strArg # path to the second frame
# end

##########################################################
backwarp_fix = [[2.0 /(INP_W/32 - 1.0), 2.0 /(INP_H/32 - 1.0)],
                [2.0 /(INP_W/16 - 1.0), 2.0 /(INP_H/16 - 1.0)],
                [2.0 /(INP_W/8 - 1.0), 2.0 /(INP_H/8 - 1.0)],
                [2.0 /(INP_W/4 - 1.0), 2.0 /(INP_H/4 - 1.0)],
                [2.0 /(INP_W/2 - 1.0), 2.0 /(INP_H/2 - 1.0)],
                [2.0 /(INP_W - 1.0), 2.0 /(INP_H - 1.0)]]
backwarp_tenGrid = {}

def backwarp(tenInput, tenFlow,intLevel):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
        tenVer = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1)
        if intLevel == 0:
			# backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHorizontal, tenVertical ], 1).permute(0, 2, 3, 1)
            backwarp_tenGrid[str(tenFlow.size())] = torch.cat([ tenHor, tenVer ], 1).permute(0, 2, 3, 1).contiguous()
	
    # end
    tenFlow_1, tenFlow_2 = tenFlow.split(split_size = 1,dim = 1)
    tenFlow = torch.cat([ tenFlow_1 * backwarp_fix[intLevel][0], tenFlow_2 * backwarp_fix[intLevel][1] ], 1)
    if intLevel == 0:
        res = torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)]), mode='bilinear', padding_mode='border', align_corners=True)
    else:
        res = torch.nn.functional.grid_sample(input=tenInput, grid=(tenFlow + backwarp_tenGrid[str(tenFlow.size())]).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=True)
    return res
    # return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=True)
# end

##########################################################

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        class Preprocess(torch.nn.Module):
            def __init__(self):
                super().__init__()
            # end

            def forward(self, tenInput):
                tenInput = tenInput.flip([1]) # RGB->BGR
                tenInput = tenInput - torch.tensor(data=[0.485, 0.456, 0.406], dtype=tenInput.dtype, device=tenInput.device).view(1, 3, 1, 1)
                tenInput = tenInput * torch.tensor(data=[1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225], dtype=tenInput.dtype, device=tenInput.device).view(1, 3, 1, 1)

                return tenInput
            # end
        # end

        class Basic(torch.nn.Module):
            def __init__(self, intLevel):
                super().__init__()

                self.netBasic = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
                )
            # end

            def forward(self, tenInput):
                return self.netBasic(tenInput)
            # end
        # end

        self.netPreprocess = Preprocess()
        self.tenUpsampled_init = torch.zeros(1, 2, int(INP_H/32), int(INP_W/32))
        self.netBasic = torch.nn.ModuleList([ Basic(intLevel) for intLevel in range(6) ])
        state_dict = torch.load(WEIGHT_PATH).items()
        self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in state_dict},strict=False)
    # end

    def forward(self, tenOne1, tenTwo1):
        nlayer=6#下采样次数


        tenOne = [ tenOne1 ]
        tenTwo = [ tenTwo1 ]

        for intLevel in range(nlayer):
            if tenOne[0].shape[2] > 32 or tenOne[0].shape[3] > 32:
                tenOne.insert(0, torch.nn.functional.avg_pool2d(input=tenOne[0], kernel_size=2, stride=2, count_include_pad=False))
                tenTwo.insert(0, torch.nn.functional.avg_pool2d(input=tenTwo[0], kernel_size=2, stride=2, count_include_pad=False))
            # end
        # end

        for intLevel in range(len(tenOne)):#从小尺度到大尺度逐层计算
            if intLevel == 0:
                tenUpsampled = self.tenUpsampled_init
                if tenUpsampled.shape[3] != tenOne[intLevel].shape[3]: tenUpsampled = torch.nn.functional.pad(input=tenUpsampled, pad=[ 0, 1, 0, 0 ], mode='replicate')

            else:
                tenUpsampled = torch.nn.functional.interpolate(input=tenFlow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

                if tenUpsampled.shape[2] != tenOne[intLevel].shape[2]: tenUpsampled = torch.nn.functional.pad(input=tenUpsampled, pad=[ 0, 0, 0, 1 ], mode='replicate')
                if tenUpsampled.shape[3] != tenOne[intLevel].shape[3]: tenUpsampled = torch.nn.functional.pad(input=tenUpsampled, pad=[ 0, 1, 0, 0 ], mode='replicate')

            tenFlow = self.netBasic[intLevel](torch.cat([ tenOne[intLevel], backwarp(tenInput=tenTwo[intLevel], tenFlow=tenUpsampled,intLevel = intLevel), tenUpsampled ], 1)) + tenUpsampled
        # end
        tenFlow = torch.nn.functional.interpolate(tenFlow, size=(INP_H, INP_W), mode='bilinear', align_corners=False)
        translation = torch.mean(tenFlow,dim=[2,3]) 
        return tenFlow,translation
    # end
# end

netNetwork = None

##########################################################

def estimate(tenOne, tenTwo):
    global netNetwork

    if netNetwork is None:
        netNetwork = Network().eval()
    # end

    assert(tenOne.shape[1] == tenTwo.shape[1])
    assert(tenOne.shape[2] == tenTwo.shape[2])

    intWidth = tenOne.shape[2]
    intHeight = tenOne.shape[1]

    # assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    # assert(intHeight == 416) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    tenPreprocessedOne = tenOne.view(1, 3, intHeight, intWidth)
    tenPreprocessedTwo = tenTwo.view(1, 3, intHeight, intWidth)

    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

    tenPreprocessedOne = torch.nn.functional.interpolate(input=tenPreprocessedOne, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
    tenPreprocessedTwo = torch.nn.functional.interpolate(input=tenPreprocessedTwo, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
    tenFirst1 =  netNetwork.netPreprocess(tenPreprocessedOne) 
    tenSecond1 =  netNetwork.netPreprocess(tenPreprocessedTwo) 
    tenFlow=netNetwork(tenFirst1, tenSecond1)[0]

    dummy_inputs = {
        "tenOne": tenFirst1,
        "tenTwo": tenSecond1,
    }
    # 先运行一遍再trace
    torch.jit.save(torch.jit.trace(netNetwork,tuple(dummy_inputs.values())),TRACED_MODEL_PATH) 


    tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

    return tenFlow[0, :, :, :].cpu()
# end

##########################################################

if __name__ == '__main__':
    img1 = PIL.Image.open(args_strOne)
    img2 = PIL.Image.open(args_strTwo)
    img1 = img1.resize([INP_W,INP_H])
    img2 = img2.resize([INP_W,INP_H])

    tenOne = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(img1)[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    tenTwo = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(img2)[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))

    tenOutput = estimate(tenOne, tenTwo)
    # objOutput = open('./out1.flo', 'wb')
    # numpy.array([ 80, 73, 69, 72 ], numpy.uint8).tofile(objOutput)
    # numpy.array([ tenOutput.shape[2], tenOutput.shape[1] ], numpy.int32).tofile(objOutput)
    # numpy.array(tenOutput.numpy().transpose(1, 2, 0), numpy.float32).tofile(objOutput)
    # objOutput.close()
    image = flow_to_image(tenOutput[ :, :, :].detach().numpy().transpose(1, 2, 0))
    plt.imshow(image)
    plt.show()

# end