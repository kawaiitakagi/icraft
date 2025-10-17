import sys
sys.path.append(R"../../../Deps/modelzoo")

import io
import os
import cv2
import yaml
import numpy as np
from tqdm import tqdm

from icraft.xir import *
from icraft.xrt import *
from icraft.buyibackend import *
from icraft.host_backend import *
from pyrtutils.icraft_utils import *
from pyrtutils.Netinfo import Netinfo
from pyrtutils.utils import *
from pyrtutils.et_device import *
from pyrtutils.calctime_utils import *

from preprocess import letterbox

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

if __name__ == '__main__':
    
    # 获取yaml
    Yaml_Path = "../cfg/nafnet_lite.yaml"
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
    if save:
        os.makedirs(resRoot, exist_ok=True)

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
        orig_H, orig_W = img_raw.shape[:2]
        if orig_H < 736:
            pad_h = 736 - orig_H
            img = np.concatenate([img_raw, np.zeros((pad_h, orig_W, 3), dtype=img_raw.dtype)], axis=0)
        elif orig_H > 736:
            img = img[:736, :, :]

        # cv2.imshow(" ", img)
        # print(img)
        # 关键分支，和 ori 保持一致
        if stage in ["a","g"] or not run_sim:
            img = img.reshape(netinfo.i_shape[0])
            # img = img.astype(np.float32).reshape(netinfo.i_shape[0])
        else:
            img = img.astype(np.float32).reshape(netinfo.i_shape[0])
        
        input_tensor = Tensor(img, Layout("NHWC"))
        # dma init(if use imk)
        dmaInit(run_sim,netinfo.ImageMake_on, netinfo.i_shape[0][1:],input_tensor, device)

        output_tensors = session.forward([input_tensor])

        if not run_sim: 
            device.reset(1)
            calctime_detail(session,network, name="./"+network.name+"_time.xlsx")

        print('shape =', np.array(output_tensors[0]).shape)
        print('INFO: get forward results!')

        res_out = np.array(output_tensors[0])
        if res_out.ndim == 4:
            res_out = np.squeeze(res_out, axis=0)
        res_out = res_out[:720, :, :]  # 只保留前720行
        res_out = np.clip(res_out, 0, 1)
        res_out = (res_out * 255.0).round().astype(np.uint8)
    
        
        if save:
            cv2.imwrite(os.path.join(resRoot, line), res_out)
        if show:
            cv2.imshow(" ", res_out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    if not run_sim: Device.Close(device)    
