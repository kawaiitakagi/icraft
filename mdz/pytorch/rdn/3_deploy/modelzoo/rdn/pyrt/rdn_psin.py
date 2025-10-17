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
from rdn_utils import *
if __name__ == "__main__":
    # 获取yaml
    Yaml_Path = "../cfg/rdn_2x.yaml"
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
    eval = cfg["imodel"]["eval"]
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

        image = pil_image.open(img_path).convert('RGB').resize([netinfo.i_shape[0][2]*2,netinfo.i_shape[0][1]*2])
        hr = image.resize((netinfo.i_shape[0][2]*2,netinfo.i_shape[0][1]*2), resample=pil_image.BICUBIC)
        lr = hr.resize((netinfo.i_shape[0][2],netinfo.i_shape[0][1]), resample=pil_image.BICUBIC)
        bicubic = lr.resize((netinfo.i_shape[0][2]*2,netinfo.i_shape[0][1]*2), resample=pil_image.BICUBIC)
        # lr = np.expand_dims(np.array(lr).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
        hr = np.expand_dims(np.array(hr).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
        hr = torch.from_numpy(hr)

        lr = np.expand_dims(np.array(lr).astype(np.float32).transpose([2, 0, 1]), 0)  #premean prescalse 已经完成

        # 构造网络输入
        if stage in ["a","g"]or not run_sim:
            lr = lr.astype(np.uint8).transpose([0,2,3,1]).copy()
            # img = img.astype(np.float32).reshape(netinfo.i_shape[0])
        else:
            lr = lr.astype(np.float32).transpose([0,2,3,1]).copy()
        input_tensor = Tensor(lr,Layout("NHWC"))

        # dma init(if use imk)
        dmaInit(run_sim,netinfo.ImageMake_on, netinfo.i_shape[0][1:],input_tensor, device)
        # run
        output_tensors = session.forward([input_tensor])
        # print(output_tensors[0])
        if not run_sim: 
            device.reset(1)
            calctime_detail(session,network, name="./"+network.name+"_time.xlsx")

        #post process

        preds_y = convert_rgb_to_y(denormalize(torch.from_numpy(np.array(output_tensors[0]).transpose([0,3,1,2])).squeeze(0)), dim_order='chw')
        hr_y = convert_rgb_to_y(denormalize(hr.squeeze(0)), dim_order='chw')
        preds_y = preds_y[2:-2, 2:-2]
        hr_y = hr_y[2:-2, 2:-2]
        psnr = calc_psnr(hr_y, preds_y)
        output = pil_image.fromarray(denormalize(torch.from_numpy(np.array(output_tensors[0]).transpose([0,3,1,2])).squeeze(0)).permute(1, 2, 0).byte().cpu().numpy())
        if show:
            output.show("rdn")

        if save:
            if not os.path.exists(resRoot):
                os.makedirs(resRoot)
            output_path = os.path.join(resRoot,line.replace('.','_result.'))
            print('result save at',output_path)
            image.save(output_path)
        if eval:
            print("The following is precision information")
            print('PSNR: {:.2f}'.format(psnr))
            print("Above is precision information")

            with open(network.name+"_metrics.log",'w+') as f:
                f.write('PSNR: {:.2f}'.format(psnr))
    if not run_sim: Device.Close(device)    
