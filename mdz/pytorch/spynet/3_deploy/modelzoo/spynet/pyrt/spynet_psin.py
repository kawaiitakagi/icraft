import sys
sys.path.append(R"../../../Deps/modelzoo")
import icraft
from icraft.xir import *
from icraft.xrt import *
from icraft.buyibackend import *
from icraft.host_backend import *
from pyrtutils.icraft_utils import *
from pyrtutils.Netinfo import Netinfo
from pyrtutils.utils import *
from pyrtutils.et_device import *
from pyrtutils.calctime_utils import *
from tqdm import tqdm
import yaml
import os
import cv2
import torchvision.transforms as transforms
import PIL.Image as pil_image
import torch
import numpy as np
from spynet_utils import *
if __name__ == "__main__":
    # 获取yaml
    Yaml_Path = "../cfg/spynet.yaml"
    if len(sys.argv) < 2:
        mprint("Info:未传入yaml参数,读入默认yaml文件:‌{}进行相关配置.".format(Yaml_Path), VERBOSE, 0)        
    if len(sys.argv) == 2:
        Yaml_Path = sys.argv[1]
        mprint("info:传入yaml文件:‌{}进行相关配置.".format(Yaml_Path), VERBOSE, 0)        
    if len(sys.argv) > 2:
        mprint("info:传入参数数量错误,请检查运行命令!", VERBOSE, 0)        
        sys.exit(1)
    # 从yaml里读入配置
    cfg = yaml.load(open(Yaml_Path, "r"), Loader=yaml.FullLoader)   
    folderPath = cfg["imodel"]["dir"]
    stage = cfg["imodel"]["stage"]
    run_sim = cfg["imodel"]["sim"]
    JSON_PATH, RAW_PATH = getJrPath(folderPath,stage,run_sim)

    load_mmu = cfg["imodel"]["mmu"]
    load_speedmode = cfg["imodel"]["speedmode"]
    load_compressFtmp = cfg["imodel"]["compressFtmp"]
    ip = str(cfg["imodel"]["ip"])
    show = cfg["imodel"]["show"]
    save = cfg["imodel"]["save"]

    imgRoot = os.path.abspath(cfg["dataset"]["dir"])
    imgList = os.path.abspath(cfg["dataset"]["list"])
    resRoot = cfg["dataset"]["res"]


    # 加载network
    network = loadNetwork(JSON_PATH, RAW_PATH)
  	# 初始化netinfo
    netinfo = Netinfo(network)
    # 选择对网络进行切分
    # network_view = network.view(netinfo.inp_shape_opid + 1)
    network_view = network.view(0)
    # 打开device
    device = openDevice(run_sim, ip, netinfo.mmu or load_mmu)
    # 初始化session
    session = initSession(run_sim, network_view, device, netinfo.mmu or load_mmu, load_speedmode, load_compressFtmp)
	#开启计时功能
    session.enableTimeProfile(True)
	#session执行前必须进行apply部署操作
    session.apply()

    for line in tqdm(open(imgList, "r")):
        name1,name2 = line.strip().split(';')
        img1_path = os.path.join(imgRoot, name1)
        img2_path = os.path.join(imgRoot, name2)
        # pre process
        img1 = pil_image.open(img1_path).convert("RGB")
        img2 = pil_image.open(img2_path).convert("RGB")
        img1 = img1.resize([netinfo.i_shape[0][2],netinfo.i_shape[0][1]])
        img2 = img2.resize([netinfo.i_shape[1][2],netinfo.i_shape[1][1]])

        tenOne = np.expand_dims(np.ascontiguousarray(np.array(img1).astype(np.float32)),axis=0).copy()
        tenTwo = np.expand_dims(np.ascontiguousarray(np.array(img2).astype(np.float32)),axis=0).copy()
        input1_tensor = Tensor(tenOne,Layout("NHWC"))
        input2_tensor = Tensor(tenTwo,Layout("NHWC"))

        # run
        output_tensors = session.forward([input1_tensor,input2_tensor])
        # print(output_tensors[0])
        # print(output_tensors[1])
        if not run_sim: 
            device.reset(1)
            calctime_detail(session,network, name="./"+network.name+"_time.xlsx")

        #post process
        tenFlow = torch.from_numpy(np.array(output_tensors[0]).transpose([0,3,1,2]))[0, :, :, :]

        if show:
            image = flow_to_image(tenFlow[ :, :, :].detach().numpy().transpose(1, 2, 0))
            plt.imshow(image)
            plt.show()
        if save:
            if not os.path.exists(resRoot):
                os.makedirs(resRoot)
            output_path = os.path.join(resRoot,name1.split('.')[0]+name2.split('.')[0]+'_result.flo')
            print('result save at',output_path)
            objOutput = open(output_path, 'wb')
            np.array([ 80, 73, 69, 72 ], np.uint8).tofile(objOutput)
            np.array([ tenFlow.shape[2], tenFlow.shape[1] ], np.int32).tofile(objOutput)
            np.array(tenFlow.numpy().transpose(1, 2, 0), np.float32).tofile(objOutput)
            objOutput.close()

    if not run_sim: Device.Close(device)    
