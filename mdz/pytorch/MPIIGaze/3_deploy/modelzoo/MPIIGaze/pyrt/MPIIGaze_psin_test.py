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
import tqdm
from datetime import timedelta
from process_mpiigaze_test import *
from pyrtutils.icraft_utils import *
from pyrtutils.Netinfo import Netinfo
from pyrtutils.utils import *
from pyrtutils.et_device import *
from pyrtutils.calctime_utils import *

if __name__ == "__main__":
    # 获取yaml
    Yaml_Path = "../cfg/MPIIGaze_test.yaml"
    if len(sys.argv) < 2:
        print("Info:未传入yaml参数,读入默认yaml文件:‌{}进行相关配置.".format(Yaml_Path), VERBOSE, 0)        
    if len(sys.argv) == 2:
        Yaml_Path = sys.argv[1]
        print("info:传入yaml文件:‌{}进行相关配置.".format(Yaml_Path), VERBOSE, 0)        
    if len(sys.argv) > 2:
        print("info:传入参数数量错误,请检查运行命令!", VERBOSE, 0)        
        sys.exit(1)
    # 从yaml里读入配置
    cfg = yaml.load(open(Yaml_Path, "r", encoding="utf-8"), Loader=yaml.FullLoader)   
    folderPath = cfg["imodel"]["net_dir"]
    stage = cfg["imodel"]["stage"]
    run_sim = cfg["imodel"]["sim"]
    JSON_PATH, RAW_PATH = getJrPath(folderPath,stage,run_sim)

    load_mmu = cfg["imodel"]["mmu"]
    load_speedmode = cfg["imodel"]["speedmode"]
    load_compressFtmp = cfg["imodel"]["compressFtmp"]
    ip = str(cfg["imodel"]["ip"])
    save = cfg["imodel"]["save"]
    calctime = cfg["imodel"]["calctime"]

    resRoot = cfg["dataset"]["res"]
    if not os.path.exists(resRoot):
        os.makedirs(resRoot)

    dataset_dir = cfg["dataset"]["dir"]

    # 加载测试集
    person_id = f'p{cfg["param"]["test_id"]:02}'
    data_dir = pathlib.Path(dataset_dir+'/Data/Normalized/')
    eval_dir = pathlib.Path(dataset_dir+'/Evaluation_Subset/sample_list_for_eye_image/')
    # 检查路径是否存在，不存在用备用路径
    if not eval_dir.exists():
        eval_dir = pathlib.Path(dataset_dir + '/Evaluation Subset/sample list for eye image/')
    images, poses, gazes = load_one_person(person_id, data_dir, eval_dir)

    
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
    

	#开启计时功能
    session.enableTimeProfile(True)

	#session执行前必须进行apply部署操作
    session.apply()
    # 精度测试
    predictions = []
    gts = []
    for i in tqdm.tqdm(range(len(images))):
        if stage in ["a","g"] and netinfo.ImageMake_on:
            image = (images[i]/255).astype(np.uint8).reshape(netinfo.i_shape[0])
            pose = poses[i].astype(np.float32).reshape(netinfo.i_shape[1])
            gaze = gazes[i].astype(np.float32).reshape(1,2) #gt
        else:
            image = (images[i]/255).astype(np.float32).reshape(netinfo.i_shape[0])
            pose = poses[i].astype(np.float32).reshape(netinfo.i_shape[1])
            gaze = gazes[i].astype(np.float32).reshape(1,2) #gt
        # 构造Icraft tensor
        inputs=[]
        inputs.append(Tensor(image, Layout("***C")))
        inputs.append(Tensor(pose, Layout("*C")))
        # dma init(if use imk)
        dmaInit(run_sim, netinfo.ImageMake_on, netinfo.i_shape[0][1:], inputs[0], device)

        # 前向
        generated_output = session.forward(inputs)
        for tensor in generated_output:
            timeout = timedelta(milliseconds=1000)
            tensor.waitForReady(timeout)

        if not run_sim:
            device.reset(1)
            if calctime:
                calctime_detail(session, network, name="./"+network.name+"_time.xlsx")

        # 后处理
        outputs = np.array(generated_output[0])
        predictions.append(torch.tensor(outputs))
        gts.append(torch.tensor(gaze))

    predictions = torch.cat(predictions)
    gts = torch.cat(gts)
    angle_error = float(compute_angle_error(predictions, gts).mean())
    print(f'The mean angle error (deg): {angle_error:.2f}')

    log_name = "MPIIGaze_metrics.log"
    logging.basicConfig(filename=log_name, level=logging.INFO, format='%(message)s')
    logging.info("angle_error: {}".format(angle_error))
    print("results save in ", log_name)
    # 关闭设备
    Device.Close(device)


    output_path = resRoot + '/predictions.npy'
    np.save(output_path, predictions.numpy())
    output_path = resRoot + '/gts.npy'
    np.save(output_path, gts.numpy())
    output_path = resRoot + '/error.txt'
    with open(output_path, 'w') as f:
        f.write(f'{angle_error}')

    
