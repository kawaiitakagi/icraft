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
from yolov5_utils import *
import torch
from post_process_yolov5 import post_detpost_soft,post_detpost_hard
if __name__ == "__main__":
    # 获取yaml
    Yaml_Path = "../cfg/yolov5s.yaml"
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
    names_path = cfg["dataset"]["names"]
    resRoot = cfg["dataset"]["res"]
    #模型自身相关参数配置
    conf = cfg["param"]["conf"]
    iou_thresh = cfg["param"]["iou_thresh"]
    multilabel = cfg["param"]["multilabel"]
    number_of_class = cfg["param"]["number_of_class"]
    anchors = cfg["param"]["anchors"]
    fpga_nms = cfg["param"]["fpga_nms"]


    # 加载network
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
        line = line.strip()
        img_path = os.path.join(imgRoot, line)
        # pre process
        img_raw = cv2.imread(img_path)
        img = letterbox(img_raw,new_shape=(netinfo.i_cubic[0].h,netinfo.i_cubic[0].w),stride=32,auto=False)[0][:,:,::-1].copy()
        if stage in ["a","g"]or not run_sim:
            img = img.reshape(netinfo.i_shape[0])
            # img = img.astype(np.float32).reshape(netinfo.i_shape[0])
        else:
            img = img.astype(np.float32).reshape(netinfo.i_shape[0])
        input_tensor = Tensor(img,Layout("NHWC"))
        # dma init(if use imk)
        dmaInit(run_sim,netinfo.ImageMake_on, netinfo.i_shape[0][1:],input_tensor, device)
        # run
        output_tensors = session.forward([input_tensor])
        # print(output_tensors[0])
        # print(output_tensors[1])
        # print(output_tensors[2])
        if not run_sim: 
            device.reset(1)
            calctime_detail(session,network, name="./"+network.name+"_time.xlsx")



        # post process
        if netinfo.DetPost_on:
            post_detpost_hard(output_tensors,img.shape,img_raw,img_path,netinfo,anchors,number_of_class,conf,iou_thresh)
        else:
            post_detpost_soft(output_tensors,img.shape,img_raw,conf,netinfo,anchors)
    
    if not run_sim: Device.Close(device)    
