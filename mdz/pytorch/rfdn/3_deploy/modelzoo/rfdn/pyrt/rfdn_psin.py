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
import numpy as np
if __name__ == "__main__":
    # 获取yaml
    Yaml_Path = "../cfg/rfdn_test.yaml"
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
    save = cfg["imodel"]["save"]
    show = cfg["imodel"]["show"]

    imgRoot = os.path.abspath(cfg["dataset"]["dir"])
    imgList = os.path.abspath(cfg["dataset"]["list"])
    resRoot = cfg["dataset"]["res"]

    # 加载network
    print('load network from',JSON_PATH)
    network = loadNetwork(JSON_PATH, RAW_PATH)
  	# 初始化netinfo
    netinfo = Netinfo(network)
    # 选择对网络进行切分
    network_view = network.view(netinfo.inp_shape_opid + 1)
    # 打开device
    device = openDevice(run_sim, ip, netinfo.mmu or load_mmu)
    # 初始化session
    session = initSession(run_sim, network_view, device, netinfo.mmu or load_mmu, load_speedmode, load_compressFtmp)
	#开启计时功能
    session.enableTimeProfile(True)
	#session执行前必须进行apply部署操作
    session.apply()



    for line in tqdm(open(imgList, "r")):
    # for line in os.listdir(imgRoot):
        line = line.strip()
        img_path = os.path.join(imgRoot, line)

        # pre process
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        print('load img dir',img_path)

        # for imk data type is unit8, but input is float32 
        if stage in ["a","g"]or not run_sim:
            img = img.reshape(netinfo.i_shape[0])        
        else:
            img = img.astype(np.float32).reshape(netinfo.i_shape[0])
        input_tensor = Tensor(img, Layout("NHWC"))
    
        # dma init(if use imk)
        dmaInit(run_sim,netinfo.ImageMake_on, netinfo.i_shape[0][1:],input_tensor, device)
        # run
        output_tensors = session.forward([input_tensor])
        device.reset(1) # must! 不重置会顺序错乱，常用reset(1)，因为较快，如果有问题就全部重置是reset(0)
        
        # time record
        if not run_sim: 
            device.reset(1)
            calctime_detail(session,network, name="./"+network.name+"_time.xlsx")

        # post process
        gen_img = np.array(output_tensors[0]).astype(np.float32)
        gen_img = np.squeeze(gen_img, axis=0)
        gen_img = cv2.cvtColor(gen_img,cv2.COLOR_RGB2BGR)
        
        if not os.path.exists(resRoot):
            os.makedirs(resRoot)

        output_path = os.path.join(resRoot, line)
        print("\n",'result save at',output_path)
        cv2.imwrite(output_path,gen_img)
    
    if not run_sim: Device.Close(device)    
