import sys
sys.path.append(R"../../../Deps/modelzoo")
from icraft.xir import *
from icraft.xrt import *
from icraft.host_backend import *
from icraft.buyibackend import *
import numpy as np
import os
import yaml
from tqdm import tqdm
from datetime import timedelta
from process_gazetr import *
from pyrtutils.icraft_utils import *
from pyrtutils.Netinfo import Netinfo
from pyrtutils.utils import *
from pyrtutils.et_device import *
from pyrtutils.calctime_utils import *

if __name__ == "__main__":
    # 获取yaml
    Yaml_Path = "../cfg/GazeTR.yaml"
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
    folderPath = cfg["imodel"]["net_dir"]
    stage = cfg["imodel"]["stage"]
    run_sim = cfg["imodel"]["sim"]
    JSON_PATH, RAW_PATH = getJrPath(folderPath,stage,run_sim)

    load_mmu = cfg["imodel"]["mmu"]
    load_speedmode = cfg["imodel"]["speedmode"]
    load_compressFtmp = cfg["imodel"]["compressFtmp"]
    ip = str(cfg["imodel"]["ip"])
    save = cfg["imodel"]["save"]
    

    resRoot = cfg["dataset"]["res"]
    if not os.path.exists(resRoot):
        os.makedirs(resRoot)

    # 加载测试图片
    imgRoot = os.path.abspath(cfg["dataset"]["dir"])
    imgList = os.path.abspath(cfg["dataset"]["list"])


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
    # 精度测试
    predictions = []

    file_list = read_list_from_txt(imgList)
    for name in tqdm(file_list):
        img_path = imgRoot + "/" + name
        img_ori = cv2.imread(img_path)
        ori_shape = img_ori.shape[:2]
        img_rgb = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)

        # 数据集的图片需要resize成模型输入尺寸
        if list(img_rgb.shape) != netinfo.i_shape[0][1:]:
            w = netinfo.i_shape[0][2]
            h = netinfo.i_shape[0][1]
            image = cv2.resize(img_rgb,(w,h))
        else:
            image = img_rgb

        if stage in ["a","g"] and netinfo.ImageMake_on:
            image = image.astype(np.uint8).reshape(netinfo.i_shape[0])
        else:
            image = image.astype(np.float32).reshape(netinfo.i_shape[0])

        # 构造Icraft tensor
        inputs=[]
        inputs.append(Tensor(image, Layout("NHWC")))
        # dma init(if use imk)
        dmaInit(run_sim, netinfo.ImageMake_on, netinfo.i_shape[0][1:], inputs[0], device)

        # 前向
        generated_output = session.forward(inputs)
        for tensor in generated_output:
            timeout = timedelta(milliseconds=1000)
            tensor.waitForReady(timeout)

        if not run_sim:
            device.reset(1)
            calctime_detail(session, network, name="./"+network.name+"_time.xlsx")

        # 后处理
        outputs = np.array(generated_output[0])
        print(f'[face] pitch: {outputs[0][1]:.2f}, yaw: {outputs[0][0]:.2f}')
        predictions.append(torch.tensor(outputs))

    predictions = torch.cat(predictions)

    output_path = resRoot + '/predictions.npy'
    np.save(output_path, predictions.numpy())

    # 关闭设备
    Device.Close(device)
    
