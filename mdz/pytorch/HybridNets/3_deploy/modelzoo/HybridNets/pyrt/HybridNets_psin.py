import sys
sys.path.append(R"../../../Deps/modelzoo")
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
from post_process_hybridnets import *
from pyrtutils.icraft_utils import *
from pyrtutils.Netinfo import Netinfo
from pyrtutils.utils import *
from pyrtutils.et_device import *
from pyrtutils.calctime_utils import *


if __name__ == "__main__":
    # 获取yaml
    Yaml_Path = "../cfg/HybridNets.yaml"
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
    MULTICLASS = cfg["param"]["multiclass"]
    threshold = cfg["param"]["threshold"]
    iou_threshold = cfg["param"]["iou_threshold"]
    seg_list = ["road","lane"]
    obj_list = ["car"]
    color_list_seg = {}
    for seg_class in seg_list:
        # edit your color here if you wanna fix to your liking
        color_list_seg[seg_class] = list(np.random.choice(range(256), size=3))

    color_list = standard_to_bgr(STANDARD_COLORS)
    anchors = torch.Tensor(np.fromfile(f'../io/anchors_384x640.ftmp',np.float32).reshape(1,46035,4))

    # 保存txt的路径
    save_dir = resRoot + "res_txt/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)


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
    if isinstance(resized_shape, list):
        resized_shape = max(resized_shape)

    for name in tqdm(file_list):
        # ======= 前处理 ===========
        img_path = imgRoot + "/" + name
        img_ori = cv2.imread(img_path)
        img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]  # orig hw
        r = resized_shape / max(h0, w0)  # resize image to img_size
        input_img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_AREA)
        h, w = input_img.shape[:2]
        (input_img, _), ratio, pad = letterbox((input_img, None), resized_shape, auto=True,
                                                scaleup=False)
        shapes = [input_img.shape[:2],((h / h0, w / w0), pad)]
        input_img = np.ascontiguousarray(input_img).reshape(1,netinfo.i_shape[0][1],netinfo.i_shape[0][2],3)
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
        out0 = output_tensors[0] # regeression
        out1 = output_tensors[1] # classification
        out2 = output_tensors[2] # seg

        regression, classification, seg = np.array(out0), np.array(out1), np.array(out2)#[1,46035,4]、[1,46035,1]、[1,384,640,3]
        seg = np.transpose(seg, (0, 3, 1, 2))#[1,3,384,640]



        res_img_path = resRoot + "/" + name
        postprocess_hybridnets(img_ori, regression, classification, seg, seg_list, obj_list, color_list, color_list_seg,anchors, pad,shapes, MULTICLASS,threshold,iou_threshold,show,save,res_img_path)


                
    # 关闭设备
    Device.Close(device)
    
