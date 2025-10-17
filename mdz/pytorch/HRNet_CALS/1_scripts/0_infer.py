import sys
import cv2
import torch
import torchvision
import numpy as np

import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms
sys.path.append(R"../0_HRNet_CALS")
sys.path.append(R"../0_HRNet_CALS/lib")

import models
from config import cfg
from config import update_config
from fp16_utils.fp16util import network_to_half
from utils.transforms import resize_align_multi_scale
from fp16_utils.fp16util import network_to_half

transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )
class args():
    def __init__(self):
        pass

def valid(img_path, cfg, model, transforms=transforms):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_size = cfg.DATASET.INPUT_SIZE
    image_resized, center, scale = resize_align_multi_scale(
        img, input_size, 1, min(cfg.TEST.SCALE_FACTOR))
    image_resized = transforms(image_resized)
    image_resized = image_resized.unsqueeze(0)
    with torch.no_grad():
        output = model(image_resized)
    return output
        
 
arg = args()
ARG_PATH = '../0_HRNet_CALS/experiments/dcs_icraft/w32_640_adam_lr1e-3_s.yaml'
WEIGHTS = '../weights/model_best.pth.tar'
img_path = R'../2_compile/qtset/DCS/000000007397.png'
arg.cfg = ARG_PATH
arg.opts = None
update_config(cfg, arg)
# load model
model = models.pose_higher_hrnet_modified.get_pose_net(cfg, is_train=False)
if cfg.FP16.ENABLED:
    model = network_to_half(model)
model.load_state_dict(torch.load(WEIGHTS,map_location=torch.device('cpu')), strict=True)
model.eval()

# get output results
output = valid(img_path, cfg, model, transforms=transforms)#[1,22,320,320]
output = output.numpy()
output = np.squeeze(output,axis=0).transpose((1,2,0))#[320,320,22]
# visualize results
heatmaps = cv2.resize(output, dsize=(1080, 1080), interpolation=cv2.INTER_LINEAR)
index_map = heatmaps.reshape(-1, 22)
ind = np.argmax(index_map, axis=0)

x = ind %1080 + 420
y = ind /1080
pred_pixel = np.vstack((x, y)).transpose()
print("pred_pixel =\n",pred_pixel)
img=cv2.imread(img_path)
for p in pred_pixel:
    cv2.circle(img, (int(p[0]-420), int(p[1])), 1, (125, 255, 0), 2)
res_path = R"infer_res.png"
cv2.imwrite(res_path, img)
print('Inference result save at:',res_path)
