import argparse
import functools
import torch 
import numpy as np
import yaml
import os 
import logging
level = logging.INFO
logging.basicConfig(level=level, format='%(asctime)s - [%(levelname)s] - %(message)s')

import sys
sys.path.append(R"../../../../0_CAM++") 
from mvector.predict import MVectorPredictor
from mvector.utils.utils import add_arguments, print_arguments

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

def feature_refiner(ori_audio_feature,max_freq_length=365,freq_size=80):
    pad_features = torch.zeros((1, max_freq_length, freq_size), dtype=torch.float32)
    if ori_audio_feature.shape[1]  < max_freq_length:#pad audio_feature
                pad_features[:, :ori_audio_feature.shape[1], :] = ori_audio_feature 
    else:# audio_feature.shape[1] >  398,crop audio_feature
        pad_features = ori_audio_feature[:,:max_freq_length,:]
    return pad_features
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
    testList = os.path.abspath(cfg["dataset"]["list"])
    resRoot = cfg["dataset"]["res"]
    #加载参数
    threshold = cfg["param"]["thr"]

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

    # 获取识别器
    predictor = MVectorPredictor(configs='configs/cam++.yml',
                                model_path='../../../../weights/CAMPPlus_Fbank/best_model/',
                                use_gpu=False)
    # 获取测试音频路径
    audio_path = []
    with open(testList,'r',encoding='utf-8') as f:
         audio_path = f.readlines()
    
    audio_path1 = audio_path[0].rstrip()
    audio_path2 = audio_path[1].rstrip()
    # 加载音频文件1，并进行预处理
    input_data_1 = predictor._load_audio(audio_data=audio_path1, sample_rate=16000)
    input_data_1 = torch.tensor(input_data_1.samples, dtype=torch.float32).unsqueeze(0)
    audio_feature_1 = predictor._audio_featurizer(input_data_1)
    print('audio_feature_1 =',audio_feature_1.shape)
    # 加载音频文件2，并进行预处理
    input_data_2 = predictor._load_audio(audio_data=audio_path2, sample_rate=16000)
    input_data_2 = torch.tensor(input_data_2.samples, dtype=torch.float32).unsqueeze(0)
    audio_feature_2 = predictor._audio_featurizer(input_data_2)
    print('audio_feature_2=',audio_feature_2.shape)
    #将音频特征统一补齐或裁剪至max_freq_length
    max_freq_length = 365
    freq_size = 80
    pad_features_1 = feature_refiner(audio_feature_1,max_freq_length,freq_size)
    pad_features_2 = feature_refiner(audio_feature_2,max_freq_length,freq_size)

    print('pad_features_1 =',pad_features_1.shape)
    print('pad_features_2=',pad_features_2.shape)
    pad_features_1 = pad_features_1.contiguous().numpy()#convert tensor to ndarray with shape (1,365,80)
    input_tensor_1 = Tensor(pad_features_1, Layout("**C"))
    pad_features_2 = pad_features_2.contiguous().numpy()#convert tensor to ndarray with shape (1,365,80)
    input_tensor_2 = Tensor(pad_features_2, Layout("**C"))

    # 模型前向推理
    output_1 = session.forward([input_tensor_1])
    if not run_sim:
        device.reset(1)
    output_2 = session.forward([input_tensor_2])
    if not run_sim:
        device.reset(1)

    feature_1 = np.array(output_1[0])[0]
    feature_2 = np.array(output_2[0])[0]
    print('INFO: get forward results!')
    print('feature_1 =',feature_1.shape)
    print('feature_2 =',feature_2.shape)

    # 对角余弦值
    dist = np.dot(feature_1, feature_2) / (np.linalg.norm(feature_1) * np.linalg.norm(feature_2))
    print(dist)

    if dist > threshold:
        print(f"{audio_path1} 和 {audio_path2} 为同一个人，相似度为：{dist}")
    else:
        print(f"{audio_path1} 和 {audio_path2} 不是同一个人，相似度为：{dist}")
    
    # saveRes
    if save:
        # save pred results 
        np.savetxt(resRoot +'/result.txt', [dist],fmt="%.6f")

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
    # Yaml_Path = "../cfg/CAM++.yaml"
    if len(sys.argv) < 2:
        print("Info:未传入yaml参数,读入默认yaml文件:‌{}进行相关配置.".format(Yaml_Path), VERBOSE, 0)        
    if len(sys.argv) == 2:
        Yaml_Path = sys.argv[1]
        print("info:传入yaml文件:‌{}进行相关配置.".format(Yaml_Path), VERBOSE, 0)        
    if len(sys.argv) > 2:
        print("info:传入参数数量错误,请检查运行命令!", VERBOSE, 0)        
        sys.exit(1)
    main(Yaml_Path)
