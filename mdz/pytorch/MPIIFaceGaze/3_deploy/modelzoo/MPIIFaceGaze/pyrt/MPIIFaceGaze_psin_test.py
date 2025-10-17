import sys
sys.path.append(R"../../../Deps/modelzoo")
from icraft.xir import *
from icraft.xrt import *
from icraft.host_backend import *
from icraft.buyibackend import *
import numpy as np
import os
import yaml
import logging
from tqdm import tqdm
from datetime import timedelta
from process_mpiifacegaze_test import *
from pyrtutils.icraft_utils import *
from pyrtutils.Netinfo import Netinfo
from pyrtutils.utils import *
from pyrtutils.et_device import *
from pyrtutils.calctime_utils import *

if __name__ == "__main__":
    # 获取yaml
    Yaml_Path = "../cfg/MPIIFaceGaze_test.yaml"
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

    # 加载测试集
    dataset_dir = pathlib.Path(cfg["dataset"]["dir"])
    person_id = f'p{cfg["param"]["test_id"]:02}'
    images, poses, gazes = add_mat_data_to_hdf5(person_id, dataset_dir)
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
    gts = []
    for i in tqdm(range(len(images))):
        # 数据集的图片需要resize成模型输入尺寸
        if images[i].shape != netinfo.i_shape[0][1:]:
            w = netinfo.i_shape[0][2]
            h = netinfo.i_shape[0][1]
            image = cv2.resize(images[i],(w,h))
        else:
            image = images[i]

        if stage in ["a","g"] and netinfo.ImageMake_on:
            image = image.astype(np.uint8).reshape(netinfo.i_shape[0])
            gaze = gazes[i].astype(np.float32).reshape(1,2) # gt
        else:
            image = image.astype(np.float32).reshape(netinfo.i_shape[0])
            gaze = gazes[i].astype(np.float32).reshape(1,2) # gt
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


        # 后处理
        outputs = np.array(generated_output[0])
        predictions.append(torch.tensor(outputs))
        gts.append(torch.tensor(gaze))

    predictions = torch.cat(predictions)
    gts = torch.cat(gts)
    angle_error = float(compute_angle_error(predictions, gts).mean())
    print(f'The mean angle error (deg): {angle_error:.2f}')

    # 打印log
    log_name = "MPIIFaceGaze_metrics.log"
    logging.basicConfig(filename=log_name, level=logging.INFO, format='%(message)s')
    logging.info("angle_error: {}".format(angle_error))
    print("results save in ", log_name)


    output_path = resRoot + '/predictions.npy'
    np.save(output_path, predictions.numpy())
    output_path = resRoot + '/gts.npy'
    np.save(output_path, gts.numpy())
    output_path = resRoot + '/error.txt'
    with open(output_path, 'w') as f:
        f.write(f'{angle_error}')
    
    # 关闭设备
    Device.Close(device)
    
