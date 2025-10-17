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
import torch
import torch.nn.functional as F
from models.yolo import *
from models.common import *

RED = '\033[31m'       # 设置前景色为红色
RESET = '\033[0m'      # 重置所有属性到默认值
ver = torch.__version__
assert ("1.6" in ver) or ("1.9" in ver), f"{RED}Unsupported PyTorch version: {ver}{RESET}"

#----------------------------
# 路径配置
#----------------------------
# 原始权重地址
W_PATH = R"../weights/yolov7-w6-pose.pt"
# trace后模型地址
TRACE_PATH = R"../2_compile/fmodel/yolov7-w6-pose-640x640.pt"

#----------------------------
# 等效reorg
#----------------------------
REORG_WEIGHTS=torch.zeros(12,3,2,2)
for i in range(12):
    j=i%3
    k=(i//3)%2
    l=0 if i<6 else 1
    REORG_WEIGHTS[i,j,k,l]=1

REORG_BIAS = torch.zeros(12)

def reorg_forward(self,x):
    y = F.conv2d(x,REORG_WEIGHTS,bias=REORG_BIAS,stride=2,padding=0,dilation=1,groups=1)
    return y


#----------------------------
# 截取IKeypoint forward：合并add和mul到conv中;去掉cpu算子;
#----------------------------
def kp_forward(self, x):
    for i in range(self.nl):
        x[i] = torch.cat((self.m[i](x[i]), self.m_kpt[i](x[i])), axis=1)
    return x 


#----------------------------
# 加载模型
#----------------------------
weigths = torch.load(W_PATH , map_location="cpu")
model = weigths['model'].float().eval()

module_list = []
for module in model.modules():
    if isinstance(module, IKeypoint):
        module_list.append(module)
# ikp = model.module.model[-1]

#----------------------------
# 优化模型
# 将IKeypoint类中ImplicitA，ImplicitM的乘和加常数的操作融合到self.m的卷积参数中
#----------------------------
model_IKeypoint = module_list[0]
head_num = len(model_IKeypoint.ia)
add_param,mul_param,last_conv_w,last_conv_b=[[0] * head_num for _ in range(4)]
for i in range(head_num):
    add_param[i]=model_IKeypoint.ia[i].implicit
    mul_param[i]=model_IKeypoint.im[i].implicit
    last_conv_w[i]=model_IKeypoint.m[i].weight.data
    last_conv_b[i]=model_IKeypoint.m[i].bias.data

#-----将ImplicitA，ImplicitM中的常数等效融合到self.m的卷积参数中
mul_param = [mul_param[i].permute(1,0,2,3).contiguous() for i in range(4)]
new_w = [last_conv_w[i]*mul_param[i] for i in range(4)]

last_conv_w = [torch.squeeze(last_conv_w[i]) for i in range(4)]
add_param=[torch.unsqueeze(torch.squeeze(add_param[i]),dim = 1) for i in range(4)]
mul_param=[torch.squeeze(mul_param[i]) for i in range(4)]
new_b = [mul_param[i]*(last_conv_b[i] + torch.squeeze(last_conv_w[i]@add_param[i])) for i in range(4)]

#-----替换其参数--------------------------
for i in range(head_num):
    model_IKeypoint.m[i].weight.data = new_w[i]
    model_IKeypoint.m[i].bias.data = new_b[i]

#----------------------------
# 替换ReOrg和IKeypoint的默认forward
#----------------------------
IKeypoint.__call__ = kp_forward
ReOrg.__call__ = reorg_forward

#----------------------------
# 添加隔离卷积（使用硬算子产生的限制）
#----------------------------
dumyin = torch.randn(1,3,640,640,dtype=torch.float32)
out = model(dumyin)
out_c = out[0].size(1)
fake_conv = nn.Conv2d(out_c,out_c,1)
fake_conv.weight.data = torch.eye(out_c).view(out_c,out_c,1,1)
fake_conv.bias.data = torch.zeros(out_c)

class New(nn.Module):
    def __init__(self,model):
        super(New, self).__init__()
        self.conv = fake_conv
        self.pre = model
    def forward(self, x):
        y = self.pre(x)
        for i in range(4):
            y[i]=self.conv(y[i])
        return y

model = New(model)
model.eval()
#----------------------------
# trace模型
#----------------------------

_=model(dumyin)
tmodel=torch.jit.trace(model,dumyin)
tmodel.save(TRACE_PATH)