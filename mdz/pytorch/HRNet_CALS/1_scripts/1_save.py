import sys
import torch
sys.path.append(R"../0_HRNet_CALS")
sys.path.append(R"../0_HRNet_CALS/lib")

import models as models
from config import cfg
from config import update_config
ARG_PATH = '../0_HRNet_CALS/experiments/dcs_icraft/w32_640_adam_lr1e-3_s.yaml'
WEIGHTS = '../weights/model_best.pth.tar'
TRACE_PATH = "../2_compile/fmodel/HRNet_640_s_icraft_traced.pt"
PLIN_TRACE_PATH = "../2_compile/fmodel/HRNet_540_s_icraft_traced.pt"
class args():
    def __init__(self):
        pass

arg = args()
arg.cfg = ARG_PATH
cfg = cfg
update_config(cfg, arg)
model = models.pose_higher_hrnet_modified.get_pose_net(cfg, is_train=False)
model.load_state_dict(torch.load(WEIGHTS, map_location=torch.device('cpu')), strict=True)
model = model.eval()

input = torch.ones((1, 3, 640, 640))
output = model(input)
print("output size =",output.size())
tmodel = torch.jit.trace(model, input)

tmodel.save(TRACE_PATH)
print('Traced model save at:',TRACE_PATH)
# trace plin model
pl_input = torch.ones((1, 3, 540, 540))
pl_model = torch.jit.trace(model,pl_input)
pl_model.save(PLIN_TRACE_PATH)
print('Traced pl_model save at:',PLIN_TRACE_PATH)