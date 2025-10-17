import sys
sys.path.append(R"../../../Deps/modelzoo")
import torch
from icraft.xir import *
from icraft.xrt import *
from icraft.host_backend import *
from icraft.buyibackend import *
import numpy as np
import cv2
import os
import yaml
from tqdm import tqdm
from datetime import timedelta
from post_process_yolop import *
from pyrtutils.icraft_utils import *
from pyrtutils.Netinfo import Netinfo
from pyrtutils.utils import *
from pyrtutils.et_device import *
from pyrtutils.calctime_utils import *


if __name__ == "__main__":
    # 获取yaml
    Yaml_Path = "../cfg/yolop.yaml"
    if len(sys.argv) < 2:
        print("Info:未传入yaml参数,读入默认yaml文件:‌{}进行相关配置.".format(Yaml_Path), VERBOSE, 0)        
    if len(sys.argv) == 2:
        Yaml_Path = sys.argv[1]
        print("info:传入yaml文件:‌{}进行相关配置.".format(Yaml_Path), VERBOSE, 0)        
    if len(sys.argv) > 2:
        print("info:传入参数数量错误,请检查运行命令!", VERBOSE, 0)        
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


    #模型自身相关参数配置
    conf = cfg["param"]["conf"]
    iou_thresh = cfg["param"]["iou_thresh"]
    multilabel =  bool(cfg["param"]["multilabel"])
    number_of_class = int(cfg["param"]["number_of_class"])
    anchors = cfg["param"]["anchors"]
    fpga_nms = bool(cfg["param"]["fpga_nms"])
    # seg_list = ["road","lane"]
    obj_list = ["car"]
    color_list_seg = {}
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(obj_list))]


    # 加载network
    network = loadNetwork(JSON_PATH, RAW_PATH)
  	# 初始化netinfo
    netinfo = Netinfo(network)
    # 选择对网络进行切分
    network_view = network.view(0)
    # 打开device
    device = openDevice(run_sim, ip, netinfo.mmu or load_mmu)
    # 初始化session
    session = initSession(run_sim, network_view, device, netinfo.mmu or load_mmu, load_speedmode, load_compressFtmp)
	# 开启计时功能
    session.enableTimeProfile(True)
	# session执行前必须进行apply部署操作
    session.apply()

    file_list = read_list_from_txt(imgList)
    resized_shape = netinfo.i_shape[0][1:3]

    for name in tqdm(file_list):
        # ======= 前处理 ===========
        img_path = imgRoot + "/" + name
        img_ori = cv2.imread(img_path)
        ori_shape = img_ori.shape[:2]
        img_rgb = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        canvas, ratio, dw, dh, new_unpad_w, new_unpad_h = resize_unscale(img_rgb, resized_shape)
        pre_messg = [ratio,dw, dh,new_unpad_w, new_unpad_h,ori_shape]
        input_img = np.ascontiguousarray(canvas).reshape(1,netinfo.i_shape[0][1],netinfo.i_shape[0][2],3)
        if stage in ["a","g"] and netinfo.ImageMake_on:
            input_img = input_img.astype(np.uint8)
        else:
            input_img = input_img.astype(np.float32)

        # 构造Icraft tensor
        inputs = []
        inputs.append(Tensor(input_img, Layout("NHWC")))

        # dma init(if use imk)
        dmaInit(run_sim, netinfo.ImageMake_on, netinfo.i_shape[0][1:], inputs[0], device)

        # net1前向
        output_tensors = session.forward(inputs)

        # 手动同步
        for tensor in output_tensors:
            timeout = timedelta(milliseconds=50000)
            tensor.waitForReady(timeout)

        if not run_sim:
            device.reset(1)
            calctime_detail(session, network, name="./"+network.name+"_time.xlsx")
        
        # 后处理
        out0 = output_tensors[0] # det_out1
        out1 = output_tensors[1] # det_out2
        out2 = output_tensors[2] # det_out3
        out3 = output_tensors[3] # da_seg_out
        out4 = output_tensors[4] # ll_seg_out
        
        det_out1, det_out2, det_out3 = np.transpose(np.array(out0), (0, 3, 1, 2)), np.transpose(np.array(out1), (0, 3, 1, 2)), np.transpose(np.array(out2), (0, 3, 1, 2))#[[1,18,80,80],[1,18,40,40],[1,18,20,20]]
        da_seg_out, ll_seg_out = np.transpose(np.array(out3), (0, 3, 1, 2)), np.transpose(np.array(out4), (0, 3, 1, 2)) # [1,2,640,640]、[1,2,640,640]
        

        res_img_path = resRoot + "/" + name
        if not netinfo.DetPost_on:
            det_outs = [torch.tensor(det_out1),torch.tensor(det_out2),torch.tensor(det_out3)]
            da_seg_out, ll_seg_out = torch.tensor(da_seg_out), torch.tensor(ll_seg_out)
            post_process_soft(img_ori,canvas,name,det_outs,da_seg_out,ll_seg_out,netinfo,anchors,number_of_class,pre_messg,resRoot,show,save)
       
                
    # 关闭设备
    Device.Close(device)
    
