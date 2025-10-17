import numpy as np 
import sys 
import cv2
import yaml
import os 

sys.path.append(R"../../../Deps/modelzoo")
from pyrtutils.icraft_utils import *
from pyrtutils.Netinfo import Netinfo
from pyrtutils.utils import *
from pyrtutils.et_device import *
from pyrtutils.calctime_utils import *

from icraft.xir import *
from icraft.xrt import *
from icraft.host_backend import *
from icraft.buyibackend import *

from utils import preprocess,get_lanes,to_array,imshow_lanes



def main(config_path) -> None:
    # 从yaml里读入配置
    
    cfg = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
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
    timeAnalysis = cfg["imodel"]['timeRes']
    imgRoot = os.path.abspath(cfg["dataset"]["dir"])
    imgList = os.path.abspath(cfg["dataset"]["list"])
    resRoot = cfg["dataset"]["res"]
   
    # 加载network
    network = loadNetwork(JSON_PATH, RAW_PATH)
    
    # 初始化netinfo
    netinfo = Netinfo(network)
    
    # 选择对网络进行切分
    network_view = network.view(netinfo.inp_shape_opid + 1)
    
    # 打开device
    device = openDevice(run_sim, ip, netinfo.mmu or load_mmu)
    print('INFO: Open Device!')
    # 初始化session
    session = initSession(run_sim, network_view, device, netinfo.mmu or load_mmu, load_speedmode, load_compressFtmp) 
    #开启计时功能
    session.enableTimeProfile(True)
    
    #session执行前必须进行apply部署操作
    session.apply()
    print('Session apply Done!')

    # Open imgList file   
    fileHandler = open  (imgList,  "r")
    # Get list of all lines in file
    listOfLines  =  fileHandler.readlines()
    for line in listOfLines:
        # prepare input
        img_path = os.path.join(imgRoot, line)
        ori_img = cv2.imread(img_path.strip())
        # pre_process
        img = preprocess(ori_img,IMG_W = netinfo.i_cubic[0].w,IMG_H=netinfo.i_cubic[0].h) #(1,288,800,3)

        input_tensor = numpy2Tensor(img,network)
        
        # dma init(if use imk)
        dmaInit(run_sim,netinfo.ImageMake_on, netinfo.i_shape[0][1:],input_tensor, device)
        
        # run
        output_tensors = session.forward([input_tensor])

        if not run_sim:
            device.reset(1)
        
        # post_process
        out = np.array(output_tensors[0])#(1,33936)
        out = out.reshape((-1,101,56,6)) # cpu reshape to target size (1,101,56,6)
        
        lanes = get_lanes(out)
        ori_lane_lists = []
        for lane in lanes:
            ori_lane = to_array(lane)
            ori_lane_lists.append(ori_lane)

        # save and show lane results
        out_file = os.path.join(resRoot, os.path.basename(img_path)).strip()
        if save:
            imshow_lanes(ori_img,ori_lane_lists,out_file=out_file)
        if show:
            result_image = cv2.imread(out_file)
            cv2.imshow('result_image',result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    # Close file
    fileHandler.close()
    # 模型时间信息统计
    if not run_sim and timeAnalysis:
        print("*"*40,"TIME RESULTS","*"*40)
        TIME_PATH = R'./time_results.xlsx'
        calctime_detail(session,network_view, name=TIME_PATH)#获取时间信息并保存时间结果
        print('Time result save at',TIME_PATH)
    if not run_sim: Device.Close(device) 

if __name__ == '__main__':
    # YAML_CONFIG_PATH = R'../cfg/TDNN.yaml'
    Yaml_Path = sys.argv[1]
    # Yaml_Path = "../cfg/UFLD.yaml"
    if len(sys.argv) < 2:
        print("Info:未传入yaml参数,读入默认yaml文件:‌{}进行相关配置.".format(Yaml_Path), VERBOSE, 0)        
    if len(sys.argv) == 2:
        Yaml_Path = sys.argv[1]
        print("info:传入yaml文件:‌{}进行相关配置.".format(Yaml_Path), VERBOSE, 0)        
    if len(sys.argv) > 2:
        print("info:传入参数数量错误,请检查运行命令!", VERBOSE, 0)        
        sys.exit(1)
    main(Yaml_Path)
    
    

    