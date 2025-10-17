import numpy as np
import cv2 
import torch 

from icraft.xir import *
from icraft.xrt import *
from icraft.host_backend import *
from icraft.buyibackend import *
from yolov9_utils import letterbox,make_anchors,dist2bbox,DFL,non_max_suppression,scale_boxes
from visualize import vis, COCO_CLASSES
# ---------------------------------参数设置---------------------------------
# 路径设置 
# 要使用不配detpost的toml文件进行编译
GENERATED_JSON_FILE = "../3_deploy/modelzoo/yolov9/imodel/8/yolov9t_parsed.json"
GENERATED_RAW_FILE = "../3_deploy/modelzoo/yolov9/imodel/8/yolov9t_parsed.raw"
IMG_PATH = "../0_yolov9/data/images/horses.jpg"

# 加载测试图像并转成icraft.Tensor
img_raw = cv2.imread(IMG_PATH)
# 改成 auto = False这样所有的输入 就是640,640
img_resize = letterbox(img_raw, (640,640), stride=32, auto=False)[0]

#要求cvread之后的结果：[不做NHWC到NCHW的转换，BGR to RGB还是要做的]  
# icraft要求输入为NHWC，这里输入格式已经是HWC，所以不需要做transpose((2,0,1)),而在PTH那边要求输入为NCHW所以需要transpose
# RGB 2 BGR的转换 需要对C维度做，
# 错误示范 im = img_resize[::-1] 是对H维度在做！
# 正确示范 img_resize[:,:,::-1]  才是对C维度在做

im = img_resize[:,:,::-1].copy() #BGR to RGB
# Img to xir.Tensor
img_ = np.expand_dims(im,axis=0)
print('img_ =',img_.shape)
input_tensor = Tensor(img_, Layout("NHWC"))

# 加载指令生成后的网络
generated_network = Network.CreateFromJsonFile(GENERATED_JSON_FILE)
generated_network.loadParamsFromFile(GENERATED_RAW_FILE)
print('INFO: Create network!')

# 创建Session
session = Session.Create([HostBackend],generated_network.view(0),[HostDevice.Default()])
session.apply()
# 模型前向推理
generated_output = session.forward([input_tensor])
# 6 out  in n,h,w,c format
# i = 0  out = (1, 80, 80, 64) ->(1, 80, 80, 80)
# i = 1  out = (1, 80, 80, 80) ->(1, 80, 80, 64)
# i = 2  out = (1, 40, 40, 64) ->(1, 40, 40, 80)
# i = 3  out = (1, 40, 40, 80) ->(1, 40, 40, 64)
# i = 4  out = (1, 20, 20, 64) ->(1, 20, 20, 80)
# i = 5  out = (1, 20, 20, 80) ->(1, 20, 20, 64)
# check outputs
for i in range(6):
    out = np.array(generated_output[i])
    print(out.shape)
print('INFO: get forward results!')

# 组装成检测结果
outputs = []
for i in range(3):
    temp1 = np.array(generated_output[2*i])
    temp2 = np.array(generated_output[2*i+1])
    # out = np.concatenate((temp1,temp2),axis=3)
    out = np.concatenate((temp2,temp1),axis=3) #这里改了！！！！
    outputs.append(torch.tensor(out.transpose((0,3,1,2))))

# 后处理 
# dfl+sigmod
reg_max = 16
nc = 80
dfl_layer = DFL(reg_max)
anchors, strides = (x.transpose(0, 1) for x in make_anchors(outputs, torch.from_numpy(np.array([8, 16, 32],dtype=np.float32)), 0.5))
box, cls = torch.cat([xi.view(1, reg_max*4+80, -1) for xi in outputs], 2).split((reg_max*4, nc), 1)
dbox = dist2bbox(dfl_layer(box), anchors.unsqueeze(0), xywh=True, dim=1) * strides
y = torch.cat((dbox, cls.sigmoid()), 1)
# postprocess
conf_thres = 0.25
iou_thres = 0.45
pred = non_max_suppression(y,
                            conf_thres,
                            iou_thres,
                            agnostic=False,
                            max_det=300,
                            classes=None)


#自己做scale box,在原图显示 
pred[0][:, :4] = scale_boxes(im.shape[0:2], pred[0][:, :4], img_raw.shape).round()
# Print results
result_image = vis(img_raw, boxes=pred[0][:,:4], scores=pred[0][:,4], cls_ids=pred[0][:,5], conf=conf_thres, class_names=COCO_CLASSES)
cv2.imshow(" ", result_image)
cv2.waitKey(0)
