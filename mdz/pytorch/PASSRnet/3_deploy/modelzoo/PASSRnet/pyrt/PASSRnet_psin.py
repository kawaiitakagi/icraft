import torch
from torchvision import transforms
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
# from datetime import timedelta
from tqdm import tqdm
from process_passrnet import *
from pyrtutils.icraft_utils import *
from pyrtutils.Netinfo import Netinfo
from pyrtutils.utils import *
from pyrtutils.et_device import *
from pyrtutils.calctime_utils import *


if __name__ == "__main__":
    # 获取yaml
    Yaml_Path = "../cfg/PASSRnet.yaml"
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

    if not os.path.exists(resRoot):
        os.makedirs(resRoot, exist_ok=True)

    # 加载network
    network = loadNetwork(JSON_PATH, RAW_PATH)
  	# 初始化netinfo
    netinfo_net = Netinfo(network)
    # 选择对网络进行切分
    network_view = network.view(0)
    # 打开device
    device = openDevice(run_sim, ip, netinfo_net.mmu or load_mmu)
    # 初始化session
    session = initSession(run_sim, network_view, device, netinfo_net.mmu or load_mmu, load_speedmode, load_compressFtmp)
	#开启计时功能
    session.enableTimeProfile(True)
	#session执行前必须进行apply部署操作
    session.apply()

    file_list = read_list_from_txt(imgList)
    psnr_list = []
    for frame_id, imf in enumerate(tqdm(file_list)):
        img_dir = imgRoot + "/lr_x4/" + imf + "/"
        #前处理
        image_lr0 = cv2.imread(os.path.join(img_dir, "lr0.png"))#left
        image_lr1 = cv2.imread(os.path.join(img_dir, "lr1.png"))#right
        gt_dir = imgRoot + "/hr/" + imf + "/"
        gt_image_hr0 = cv2.imread(os.path.join(gt_dir, "hr0.png"))#left
        gt_image_hr1 = cv2.imread(os.path.join(gt_dir, "hr1.png"))#right

        img_lr0 = cv2.cvtColor(image_lr0, cv2.COLOR_BGR2RGB)
        img_lr1 = cv2.cvtColor(image_lr1, cv2.COLOR_BGR2RGB)
        gt_img_hr0 = cv2.cvtColor(gt_image_hr0, cv2.COLOR_BGR2RGB)
        gt_img_hr1 = cv2.cvtColor(gt_image_hr1, cv2.COLOR_BGR2RGB)
        # img_lr0 = img_lr0/255 #parse阶段配置归一化
        # img_lr1 = img_lr1/255 #parse阶段配置归一化
        gt_img_hr0 = gt_img_hr0/255
        gt_img_hr1 = gt_img_hr1/255
        gt_left_hr = torch.tensor(gt_img_hr0).unsqueeze(0).permute(0,3,1,2)
        if stage in ["a","g"] and netinfo_net.ImageMake_on:
            left_lr = img_lr0.reshape(netinfo_net.i_shape[0])
            right_lr = img_lr1.reshape(netinfo_net.i_shape[0])
        else:
            left_lr = img_lr0.reshape(netinfo_net.i_shape[0]).astype(np.float32)
            right_lr = img_lr1.reshape(netinfo_net.i_shape[0]).astype(np.float32)
        # 构造Icraft tensor
        inputs=[]
        inputs.append(Tensor(left_lr, Layout("NHWC")))# NHWC输入
        inputs.append(Tensor(right_lr, Layout("NHWC")))# NHWC输入

        # dma init(if use imk)
        dmaInit(run_sim, netinfo_net.ImageMake_on, netinfo_net.i_shape[0][1:], inputs[0], device)

        # 前向
        output_tensors = session.forward(inputs)
        if not run_sim:
            device.reset(1)
            calctime_detail(session,network, name="./"+network.name+"_time.xlsx")
        SR_left = np.array(output_tensors[0]).astype(np.float32)
        SR_left = torch.tensor(SR_left).permute(0,3,1,2)

        # 计算峰值信噪比
        SR_left = torch.clamp(SR_left, 0, 1)#限制张量中每个元素的范围[0,1]
        psnr_list.append(cal_psnr(gt_left_hr[:,:,:,64:], SR_left[:,:,:,64:]))
        print('mean psnr: ', float(np.array(psnr_list).mean()))

        SR_left_img = transforms.ToPILImage()(torch.squeeze(SR_left.data.cpu(), 0))
        
        if show:
            # 显示图像
            SR_left_img.show()
        if save:
            img_resRoot = resRoot +'/'+imf
            if not os.path.exists(img_resRoot):
                os.makedirs(img_resRoot, exist_ok=True)
            SR_left_img.save(img_resRoot+'/img_0.png')
            print("result save in ",img_resRoot+'/img_0.png')

    # 关闭设备
    Device.Close(device)
    
