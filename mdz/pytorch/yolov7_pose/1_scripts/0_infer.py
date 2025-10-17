"""
 Copyright 2023  FMSH.Co.Ltd. All rights reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """
import sys
sys.path.append(R"../0_yolov7_pose")
import matplotlib.pyplot as plt
import torch
import cv2
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression
from utils.plots import output_to_target,plot_skeleton_kpts

#----------------------------
# 路径配置
#----------------------------
# 原始权重地址
W_PATH = R"../weights/yolov7-w6-pose.pt"
# 输入图片地址
IMG_PATH = R'../0_yolov7_pose/onnx_inference/img.png'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weigths = torch.load(W_PATH, map_location=device)
model = weigths['model']
_ = model.float().eval()
if torch.cuda.is_available():
    model.half().to(device)

image = cv2.imread(IMG_PATH)
image = letterbox(image, 640, stride=64, auto=True)[0]
image_ = image.copy()
image = transforms.ToTensor()(image)
image = torch.tensor(np.array([image.numpy()]))

if torch.cuda.is_available():
    image = image.half().to(device)   
output, _ = model(image)

output = non_max_suppression(output, 0.25, 0.65, nc=1, nkpt=17, kpt_label=True)
with torch.no_grad():
    output = output_to_target(output)
nimg = image[0].permute(1, 2, 0) * 255
nimg = nimg.cpu().numpy().astype(np.uint8)
nimg = cv2.cvtColor(nimg, 0)
for idx in range(output.shape[0]):
    plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

cv2.imshow(" ",nimg)
cv2.waitKey(0)