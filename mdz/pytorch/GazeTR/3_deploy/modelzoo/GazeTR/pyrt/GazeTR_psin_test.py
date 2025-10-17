import sys
sys.path.append(R"../../../Deps/modelzoo")
import torch
from icraft.xir import *
from icraft.xrt import *
from icraft.host_backend import *
from icraft.buyibackend import *
import numpy as np
import os
import yaml
sys.path.insert(0,R"../../../../0_GazeTR")
from easydict import EasyDict as edict
import importlib
import ctools, gtools
import progressbar
import logging
from pyrtutils.icraft_utils import *
from pyrtutils.Netinfo import Netinfo
from pyrtutils.utils import *
from pyrtutils.et_device import *
from pyrtutils.calctime_utils import *



if __name__ == "__main__":
    conf = edict(yaml.load(open("./config/config_mpii_test.yaml"), Loader=yaml.FullLoader))
    test_conf = conf.test

    # 获取yaml
    Yaml_Path = "../cfg/GazeTR_test.yaml"
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
    show = cfg["imodel"]["show"]

    dataRoot = os.path.abspath(cfg["dataset"]["dir"])
    resRoot = cfg["dataset"]["res"]

    test_conf.data.image = os.path.join(dataRoot, "Image")
    test_conf.data.label = os.path.join(dataRoot, "Label")


    #加载数据集
    reader = importlib.import_module("reader.reader")
    test_id = int(cfg["param"]["test_id"])
    test_conf.person = test_id
    data = test_conf.data
    load = test_conf.load
    data, folder = ctools.readfolder(data, [test_conf.person])
    testname = folder[test_conf.person] 
    dataset = reader.loader(data, 1, num_workers=1, shuffle=True)# batch = 1
    logpath = "./log/"
    if not os.path.exists(logpath):
        os.makedirs(logpath)
    
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

    begin = load.begin_step; end = load.end_step; step = load.steps

    for saveiter in range(begin, end+step, step):
        print(f"Test {saveiter}") 
        length = len(dataset); accs = 0; count = 0
        # log
        logname = f"{saveiter}.log"
        outfile =  open(os.path.join(logpath, logname), 'w')
        outfile.write("name results gts\n")
        predictions = []
        gts = []
        img_names = []
        bar = progressbar.ProgressBar(max_value=len(dataset))
        for j, (data, label) in enumerate(dataset):
            bar.update(j)
            img_names.append(data["name"])
            gts.append(label)
            img = np.array(data["face"]).transpose(0,2,3,1)#[1,224,224,3]
            img = (img*255)[:,:,:,::-1] #数据集已做归一化，此处抵消Icraft归一化操作
            img = np.ascontiguousarray(img)
            

            if stage in ["a","g"] and netinfo.ImageMake_on:
                image = img.astype(np.uint8).reshape(netinfo.i_shape[0])
            else:
                image = img.astype(np.float32).reshape(netinfo.i_shape[0])

            # 构造Icraft tensor
            inputs=[]
            inputs.append(Tensor(image, Layout("NHWC")))

            # dma init(if use imk)
            dmaInit(run_sim, netinfo.ImageMake_on, netinfo.i_shape[0][1:], inputs[0], device)

            # 前向
            generated_output = session.forward(inputs)

            if not run_sim:
                device.reset(1)
                # calctime_detail(session, network, name="./"+network.name+"_time.xlsx")

            # 后处理
            outputs = np.array(generated_output[0])
            predictions.append(torch.tensor(outputs))
        
        predictions = torch.cat(predictions)
        gts = torch.cat(gts)
        for k, gaze in enumerate(predictions):
            gaze = gaze.cpu().detach().numpy()
            gt = np.array(gts)[k]

            count += 1
            accs += gtools.angular(
                        gtools.gazeto3d(gaze), 
                        gtools.gazeto3d(gt)
                    )
    
            name = [img_names[k]]
            gaze = [str(u) for u in gaze] 
            gt = [str(u) for u in gt] 
            log = name[0] + [",".join(gaze)] + [",".join(gt)]
            outfile.write(" ".join(log) + "\n")

        loger = f"[{saveiter}] Total Num: {count}, avg: {accs/count}"
        print(loger)
        outfile.write(loger)
        #打印精度log
        log_name = "GazeTR_metrics.log"
        logging.basicConfig(filename=log_name, level=logging.INFO, format='%(message)s')
        logging.info("angle_error: {}".format(accs/count))
        print("metric results save in ", log_name)

    outfile.close()
        
    # 关闭设备
    Device.Close(device)
    
