# 输入为off原始数据，验证j&r仿真正确性
import torch 
import numpy as np 
from icraft.xir import *
from icraft.xrt import *
from icraft.host_backend import *
from icraft.buyibackend import *
from datetime import datetime
import sys 
sys.path.append(R'../0_PointNet')
from utils.data import get_single_data
CLASS= {'airplane': 0, 'bathtub': 1, 'bed': 2, 'bench': 3, 'bookshelf': 4, 'bottle': 5, 'bowl': 6, 'car': 7, 'chair': 8, 'cone': 9, 'cup': 10, 'curtain': 11, 'desk': 12, 'door': 13, 'dresser': 14, 'flower_pot': 15, 'glass_box': 16, 'guitar': 17, 'keyboard': 18, 'lamp': 19, 'laptop': 20, 'mantel': 21, 'monitor': 22, 'night_stand': 23, 'person': 24, 'piano': 25, 'plant': 26, 'radio': 27, 'range_hood': 28, 'sink': 29, 'sofa': 30, 'stairs': 31, 'stool': 32, 'table': 33, 'tent': 34, 'toilet': 35, 'tv_stand': 36, 'vase': 37, 'wardrobe': 38, 'xbox': 39}
# ---------------------------------参数设置---------------------------------
STAGE = 'g'
stage_list = {
    'p':'parsed',
    'o':'optimized',
    'q':'quantized',
    'a':'adapted',
    'g':'BY',
}
# 路径设置 
GENERATED_JSON_FILE = "../3_deploy/modelzoo/PointNet/imodel/8/pointnet_"+stage_list[STAGE]+".json"
GENERATED_RAW_FILE  = "../3_deploy/modelzoo/PointNet/imodel/8/pointnet_"+stage_list[STAGE]+".raw"
CLASS= {'airplane': 0, 'bathtub': 1, 'bed': 2, 'bench': 3, 'bookshelf': 4, 'bottle': 5, 'bowl': 6, 'car': 7, 'chair': 8, 'cone': 9, 'cup': 10, 'curtain': 11, 'desk': 12, 'door': 13, 'dresser': 14, 'flower_pot': 15, 'glass_box': 16, 'guitar': 17, 'keyboard': 18, 'lamp': 19, 'laptop': 20, 'mantel': 21, 'monitor': 22, 'night_stand': 23, 'person': 24, 'piano': 25, 'plant': 26, 'radio': 27, 'range_hood': 28, 'sink': 29, 'sofa': 30, 'stairs': 31, 'stool': 32, 'table': 33, 'tent': 34, 'toilet': 35, 'tv_stand': 36, 'vase': 37, 'wardrobe': 38, 'xbox': 39}
data_path = R"./Data/ModelNet40/airplane/test/airplane_0627.off"

# convert pointcloud data to Icraft Tensor
_data = get_single_data(data_path) # [1024,3] np.float64
_inputs = _data.unsqueeze(0).transpose(1,2).numpy().astype(np.float32) # [1,3,1024],np.float32
_inputs = np.ascontiguousarray(_inputs)
input_tensor = Tensor(_inputs, Layout("**C"))

# 加载指令生成后的网络
generated_network = Network.CreateFromJsonFile(GENERATED_JSON_FILE)
generated_network.loadParamsFromFile(GENERATED_RAW_FILE)
print('INFO: Create network!')

# 创建Session
session = Session.Create([HostBackend],generated_network.view(0),[HostDevice.Default()])
session.apply()
# 模型前向推理
generated_output = session.forward([input_tensor])
# check outputs
# [1,40] : pred_results 
# [3,3]  : transform_matrix_3x3
# [64,64]: transform_matrix_64x64
# for i in range(3):
#     out = np.array(generated_output[i])
#     print(out.shape)
print('INFO: get forward results!')

# 组装成检测结果
outputs = []
for i in range(3):
    out = np.array(generated_output[i])
    print(out.shape)
    outputs.append(torch.tensor(out))
print("_output =",outputs[0].shape)
# 后处理

_, pred = torch.max(outputs[0], 1) # value,index=pred
pred = pred.cpu().numpy()[0]
for category in CLASS.keys():
    if CLASS[category] == pred:
        break
print(f"TEST RESULT:\n\tcategory =",category)