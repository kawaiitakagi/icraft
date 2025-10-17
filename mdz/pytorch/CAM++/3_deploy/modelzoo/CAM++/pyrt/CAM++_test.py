import argparse
import functools
import torch 
import numpy as np
import yaml
import os 
from tqdm import tqdm
import time
from sklearn.metrics.pairwise import cosine_similarity

import logging
level = logging.INFO
logging.basicConfig(level=level, format='%(asctime)s - [%(levelname)s] - %(message)s')
from torch.utils.data import DataLoader
import sys
sys.path.append(R"../../../../0_CAM++") 
from mvector.predict import MVectorPredictor
from mvector.data_utils.reader import MVectorDataset
from mvector.data_utils.featurizer import AudioFeaturizer
from mvector.data_utils.collate_fn import collate_fn
from mvector.metric.metrics import compute_fnr_fpr, compute_eer, compute_dcf, accuracy
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
    enrollRoot = os.path.abspath(cfg["dataset"]["dir"])
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
    
    # setup dataloader获取评估的检验数据
    # 获取特征器
    audio_featurizer = AudioFeaturizer(feature_method='Fbank',
                                                use_hf_model=False,
                                                method_args={'sample_frequency': 16000, 'num_mel_bins': 80})
    
    trials_dataset = MVectorDataset(data_list_path=testList,
                                    audio_featurizer=audio_featurizer,
                                    do_vad=False,
                                    max_duration=20,
                                    min_duration=0.5,
                                    sample_rate=16000,
                                    use_dB_normalization=True,
                                    target_dB=-20,
                                    mode='eval')
    print(trials_dataset)

    trials_loader = DataLoader(dataset=trials_dataset,
                                collate_fn=collate_fn,
                                shuffle=False,
                                batch_size=1,
                                num_workers=1)

    print(trials_loader)
    # 获取检验的声纹特征和标签
    count = 0
    trials_features, trials_labels = None, None
    start = time.time()
    for batch_id, (audio_features, label, input_lens) in enumerate(
                    tqdm(trials_loader, desc="验证音频声纹特征")):
        
        # get label
        label = label.long()
        label = label.data.cpu().numpy().astype(np.int32)
        # get feature
        # t_span = audio_features.shape[1],将audio_features t_span维度统一补齐到max_freq_length维度
        max_freq_length = 365
        if audio_features.shape[1] != max_freq_length:
            print(f'batch_id ={batch_id}音频 audio_feature为{audio_features.shape},不满足shape = (1,{max_freq_length},80)的要求,进行了补齐或裁剪')
            count += 1
            #因此补齐或裁剪至max_freq_length
            freq_size = 80
            pad_features = torch.zeros((1, max_freq_length, freq_size), dtype=torch.float32)
            if audio_features.shape[1]  < max_freq_length:#pad audio_feature
                pad_features[:, :audio_features.shape[1], :] = audio_features
                
            else:# audio_feature.shape[1] >  max_freq_length,crop audio_feature
                pad_features = audio_features[:,:max_freq_length,:]
            audio_features = pad_features
            print('After pad, audio features =',audio_features.shape)
        
        audio_features = audio_features.contiguous().numpy()#(1,max_freq_length,80)
        #只对输入为[1,max_freq_length,80]的audio_feature计算精度
        input_tensor = Tensor(audio_features, Layout("**C"))
        # 模型前向推理
        output = session.forward([input_tensor])
        # 重置设备
        if not run_sim: device.reset(1)
        # check outputs
        # [1,192] : pred_results = [1,N_CLASS]
        feature = np.array(output[0])
        print('INFO: get forward results!')
        feature = torch.from_numpy(feature)
        print('feature = ',feature.shape)
        # 存放特征
        trials_features = np.concatenate((trials_features, feature)) if trials_features is not None else feature
        trials_labels = np.concatenate((trials_labels, label)) if trials_labels is not None else label
        
    print('trials_features =',trials_features.shape)
    print('trials_labels =',trials_labels.shape)
    print(f'There are {count} samples time-steps != {max_freq_length}')# 17777个sample中多少个不等于max_freq_length

    # load enroll results 获取注册的声纹特征和标签
    enroll_features = np.loadtxt(enrollRoot + './enroll_features.txt')
    enroll_labels = np.loadtxt(enrollRoot + './enroll_labels.txt')
    
    print('开始对比音频特征...')  
    
    all_score, all_labels = [], []
    for i in tqdm(range(len(trials_features)), desc='特征对比'):
        trials_feature = np.expand_dims(trials_features[i], 0)
        score = cosine_similarity(trials_feature, enroll_features).astype(np.float32).tolist()[0]
        trials_label = np.expand_dims(trials_labels[i], 0).repeat(len(enroll_features), axis=0)
        y_true = np.array(enroll_labels == trials_label).astype(np.int32).tolist()
        all_score.extend(score)
        all_labels.extend(y_true)
    # 计算EER
    all_score = np.array(all_score, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int32)
    fnr, fpr, thresholds = compute_fnr_fpr(all_score, all_labels)
    eer, threshold = compute_eer(fnr, fpr, all_score)
    min_dcf = compute_dcf(fnr, fpr)
    eer, min_dcf, threshold = float(eer), float(min_dcf), float(threshold)
    end = time.time()
    if save:
        import matplotlib.pyplot as plt
        index = np.where(np.array(thresholds) == threshold)[0][0]
        plt.plot(thresholds, fnr, color='blue', linestyle='-', label='fnr')
        plt.plot(thresholds, fpr, color='red', linestyle='-', label='fpr')
        plt.plot(threshold, fpr[index], 'ro-')
        plt.text(threshold, fpr[index], (round(threshold, 3), round(fpr[index], 5)), color='red')
        plt.xlabel('threshold')
        plt.title('fnr and fpr')
        plt.grid(True)  # 显示网格线
        # 保存图像
        os.makedirs(resRoot, exist_ok=True)
        plt.savefig(os.path.join(resRoot, 'result.png'))
        logging.info(f"结果图保存在：{os.path.join(resRoot, 'result.png')}")
        # save test results
        ERR_PATH = R'./CAM++_metrics.log'
        with open (ERR_PATH, 'w') as f:
            f.write(f"EER: {eer * 100}%, MinDCF: {min_dcf}")
        print('EER result save at', ERR_PATH)
    print('评估消耗时间：{}s, threshold:{:.2f}, EER: {:.5f}, MinDCF: {:.5f}'.format(int(end - start), threshold, eer, min_dcf))
    
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
