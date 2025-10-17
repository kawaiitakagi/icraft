"""
@Description: Main script for testing PointNet accuracy on ModelNet40 dataset
"""

# %% Import packages
import numpy as np
import torch
import sys
import yaml
import os 
import logging
level = logging.INFO
logging.basicConfig(level=level, format='%(asctime)s - [%(levelname)s] - %(message)s')
# Import Icraft packages
import icraft
from icraft.xir import *
from icraft.xrt import *
from icraft.host_backend import *
from icraft.buyibackend import *
# Import pyrt dependencies
sys.path.append(R"../../../Deps/modelzoo")
from pyrtutils.Netinfo import Netinfo
from pyrtutils.utils import *
from pyrtutils.et_device import *
from pyrtutils.icraft_utils import *

# Import preprocess post_process utils
from pointnet_utils import *

def main(config_path) -> None:
    # %% set configurations
    # ---------------------------------参数设置---------------------------------
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

    imgRoot = os.path.abspath(cfg["dataset"]["dir"])
    imgList = os.path.abspath(cfg["dataset"]["list"])
    names_path = cfg["dataset"]["names"]
    resRoot = cfg["dataset"]["res"]
    metricRes = cfg["imodel"]['metricRes']
    #模型自身相关参数配置
    number_of_class = cfg["param"]["number_of_class"]
    
    
    
    # 加载network
    network = loadNetwork(JSON_PATH, RAW_PATH)
    print('INFO: Load network!')
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
    print('INFO: Session Apply!')

    logging.info(
        f"DATA INFO:\n"
        f"\tData Root Path:                {imgRoot}\n"
        "\n"

        f"Selected Computing Device:\t       {'cpu'}\n"
        "\n"
        f"Simulation Mode:\t                 {run_sim}\n"
        "\n"

        f"TEST INFO:\n"
        f"\tTested Model Path:            {JSON_PATH}\n"
        "\n"
    )
    # %% what kinds of classes do we have?
    classes = CLASS
    logging.info(f"Classes: {classes}")

    # %% data loaders
    logging.info("Loading the data...")
    test_loader = get_dataloader(
        data_path=imgRoot,
        folder="test",
        dataset_type="test",
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        shuffle=False
    )
    # %% let's test the trained network
    logging.info("Testing started...")
    all_preds = []
    all_labels = []
    correct_sum = 0
    if save:
        os.makedirs(resRoot, exist_ok=True)
    for idx, data in enumerate(test_loader):
        inputs = data['pointcloud'].float() #[1,1024,3]
        labels = data['category']
        _inputs = inputs.transpose(1,2).numpy()
        _inputs = np.ascontiguousarray(_inputs)
        # ndarray to icraft.Tensor
        input_tensor = Tensor(_inputs, Layout("**C"))
        # 模型前向推理
        generated_output = session.forward([input_tensor])
        # 前向完，重置device
        if not run_sim: 
            device.reset(1)
        # check outputs
        # [1,40] : pred_results  用(outputs[0])
        # [3,3]  : transform_matrix_3x3
        # [64,64]: transform_matrix_64x64
        # for i in range(3):
        #     out = np.array(generated_output[i])
        #     print(out.shape)
        # print('INFO: get forward results!')
        # 组装成检测结果
        outputs = []
        for i in range(3):
            out = np.array(generated_output[i])
            # print(out.shape)
            outputs.append(torch.tensor(out))
        # 后处理
        _, pred = torch.max(outputs[0], 1) # value,index=pred
        
        print(f'TEST RESULT:\n\tpred =',pred.cpu().numpy() == labels.cpu().numpy())
        # 精度计算
        all_preds += list(pred.cpu().numpy())
        all_labels += list(labels.cpu().numpy())
        correct_sum += (pred.cpu().numpy() == labels.cpu().numpy()).sum()
        # 保存pred结果为txt文件
        np.savetxt(resRoot +'/'+ str(idx)+'_result.txt', pred.cpu().numpy(),fmt="%d")
        pred = pred.cpu().numpy()[0] #np.int64
        for category in CLASS.keys():
            if CLASS[category] == pred:
                break
        
        print(f"\tcategory =",category)
        
    test_acc = correct_sum / len(all_preds) * 100.0
    sResult = f'Top-1: {test_acc}%\n'
    
    # if metricRes, save acc to metric.log
    if metricRes:
        
        # save acc to metric.log
        ACC_PATH = R"./PointNet_metrics.log"
        with open (ACC_PATH,'w') as f:
            f.write(sResult)
            f.close() 
    print('Top-1: %5.1f %% \nPred results save at %s'% ( test_acc,resRoot))  
    logging.info("Testing DONE!")
    # 关闭device
    if not run_sim: 
        Device.Close(device) 
   

if __name__ == '__main__':
    # YAML_CONFIG_PATH = R'../cfg/pointnet_test.yaml'
    Yaml_Path = sys.argv[1]
    # Yaml_Path = "../cfg/pointnet_test_tt.yaml"
    if len(sys.argv) < 2:
        print("Info:未传入yaml参数,读入默认yaml文件:‌{}进行相关配置.".format(Yaml_Path), VERBOSE, 0)        
    if len(sys.argv) == 2:
        Yaml_Path = sys.argv[1]
        print("info:传入yaml文件:‌{}进行相关配置.".format(Yaml_Path), VERBOSE, 0)        
    if len(sys.argv) > 2:
        print("info:传入参数数量错误,请检查运行命令!", VERBOSE, 0)        
        sys.exit(1)
    main(Yaml_Path)
