from icraft.xir import *
from icraft.xrt import *
from icraft.host_backend import *
from icraft.buyibackend import *
import numpy as np
from typing import List
import cv2
import os

def run(network: Network, input: List[Tensor]) -> List[Tensor]:
    session = Session.Create([ HostBackend], network.view(0), [HostDevice.Default()])
    session.apply()
    output_tensors = session.forward( input ) #前向
    return output_tensors

height   = 720
width    = 1280

dst_dir = '../3_deploy/modelzoo/mimo-unet/io'
if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)
GENERATED_JSON_FILE = "../3_deploy/modelzoo/mimo-unet/imodel/8/mimo-unet_720x1280_parsed.json"
GENERATED_RAW_FILE = "../3_deploy/modelzoo/mimo-unet/imodel/8/mimo-unet_720x1280_parsed.raw"


# 加载指令生成后的网络
generated_network = Network.CreateFromJsonFile(GENERATED_JSON_FILE)
generated_network.loadParamsFromFile(GENERATED_RAW_FILE)

img_path=(R"../3_deploy/modelzoo/mimo-unet/io/input/GOPR0384_11_00-000001.png")
img = cv2.imread(img_path)
# img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
img_=np.array(img).astype(np.float32).reshape(1, height, width, 3) 

input_tensor = Tensor(img_, Layout("NHWC"))
try:
    generated_output = run(generated_network, [input_tensor])
except InternalError as i:
    print(i)

print(np.array(generated_output[2]).shape)

gen_img = np.array(generated_output[2]).astype(np.float32)
gen_img = np.squeeze(gen_img, axis=0)
gen_img = (gen_img*255).round()
dst_path = dst_dir + '/sim_parsed.png'
cv2.imwrite(dst_path,gen_img)
