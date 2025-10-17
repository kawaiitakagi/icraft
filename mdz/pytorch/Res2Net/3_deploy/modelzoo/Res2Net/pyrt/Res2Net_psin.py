import torch 
import numpy as np
import yaml
import os 
import sys 
import logging
level = logging.INFO
logging.basicConfig(level=level, format='%(asctime)s - [%(levelname)s] - %(message)s')
from torch.utils.data import DataLoader
from tqdm import tqdm

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

sys.path.append(R"../../../../0_Res2Net")
from macls.data_utils.reader import MAClsDataset
from macls.data_utils.featurizer import AudioFeaturizer
from macls.data_utils.collate_fn import collate_fn
def get_CLASS(names_path)->list:
    class_list = list()
    if os.path.exists(names_path):
        f = open(names_path, "r")
        res = f.readlines()
        for item in res:
            class_list.append(item.strip('\n'))
        f.close()
    else:
        print(f"文件 '{names_path}' 不存在。")
    return class_list

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
    
    names_path = cfg["dataset"]["names"]
    resRoot = cfg["dataset"]["res"]

    classes = get_CLASS(names_path)
    
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
    
    logging.info(f"Classes: {classes}")
    # 获取训练集-测试数据
    audio_featurizer = AudioFeaturizer(feature_method='Fbank',
                                    method_args={'sample_frequency':16000,'num_mel_bins':80})
    test_dataset = MAClsDataset(data_list_path=testList,
                                    audio_featurizer=audio_featurizer,
                                    do_vad=False,
                                    max_duration=10,
                                    min_duration=0.5,
                                    sample_rate=16000,
                                    use_dB_normalization=True,
                                    target_dB=-20,
                                    mode='eval')
    test_loader = DataLoader(dataset=test_dataset,
                        collate_fn=collate_fn,
                        shuffle=False,
                        batch_size=1,
                        num_workers=0)
    count = 0
    accuracies = []
    for batch_id, (audio_feature, gt_label, input_lens) in enumerate(tqdm(test_loader)):
        
        gt_label = int(gt_label)
        if audio_feature.shape[1] != 398:
            # 表示音频长度不满足要求，可能太短 <0.5 s 
            print(f'batch_id ={batch_id}音频 audio_feature为{audio_feature.shape},不满足shape = (1,398,80)的要求,进行了补齐或裁剪')
            count += 1
            #因此补齐或裁剪至max_freq_length
            max_freq_length = 398
            freq_size = 80
            pad_features = torch.zeros((1, max_freq_length, freq_size), dtype=torch.float32)
            if audio_feature.shape[1]  < 398:#pad audio_feature
                pad_features[:, :audio_feature.shape[1], :] = audio_feature 
                
            else:# audio_feature.shape[1] >  398,crop audio_feature
                pad_features = audio_feature[:,:max_freq_length,:]
            audio_feature = pad_features
        
        audio_feature = audio_feature.contiguous().numpy()#(1,398,80)
        #只对输入为[1,398,80]的audio_feature计算精度
        input_tensor = Tensor(audio_feature, Layout("**C"))
        # 模型前向推理
        output = session.forward([input_tensor])
        # 重置设备
        if not run_sim: device.reset(1)
        # check outputs
        # [1,10] : pred_results = [1,N_CLASS]
        result = np.array(output[0])
        # print('INFO: get forward results!')
        result = torch.from_numpy(result)
        result = torch.nn.functional.softmax(result, dim=-1)[0]
        result = result.data.cpu().numpy()
        pred_lab = np.argsort(result)[-1]
        
        score = result[pred_lab]

        pred_label = classes[pred_lab]
        score = round(float(score), 5)
        print(f'batch_id ={batch_id}的音频预测结果标签为：{pred_label}, 得分：{score}')
        acc = np.mean((pred_lab == gt_label).astype(int))
        accuracies.append(acc)
        if save:
            # save pred results 
            os.makedirs(resRoot, exist_ok=True)
            np.savetxt(resRoot +'/'+ str(batch_id)+'_result.txt', [pred_lab],fmt="%d")
    test_acc = float(sum(accuracies) / len(accuracies)) if len(accuracies) > 0 else -1
    sResult = f"Top-1 : {test_acc * 100}%"
    if show:
        # show test accuracy results
        print("*"*40,"TEST SUMMARY","*"*40)
        print(f'There are {count} samples time-steps != 398')# 874个sample 146个不等于389
        print(sResult)
    if save:
        # save test accuracy results
        ACC_PATH = R'./Res2Net_metrics.log'
        with open (ACC_PATH,"w") as f:
            f.write(sResult)
            f.close()
        print('Accuracy result save at',ACC_PATH)
        # 模型时间信息统计
    if not run_sim and timeAnalysis:
        print("*"*40,"TIME RESULTS","*"*40)
        TIME_PATH = R'./time_results.xlsx'
        calctime_detail(session,network_view, name=TIME_PATH)#获取时间信息并保存时间结果
        print('Time result save at',TIME_PATH)

    # 关闭设备
    Device.Close(device) 
    

if __name__ == '__main__':
    # YAML_CONFIG_PATH = R'../cfg/TDNN.yaml'
    Yaml_Path = sys.argv[1]
    # Yaml_Path = "../cfg/TDNN.yaml"
    if len(sys.argv) < 2:
        print("Info:未传入yaml参数,读入默认yaml文件:‌{}进行相关配置.".format(Yaml_Path), VERBOSE, 0)        
    if len(sys.argv) == 2:
        Yaml_Path = sys.argv[1]
        print("info:传入yaml文件:‌{}进行相关配置.".format(Yaml_Path), VERBOSE, 0)        
    if len(sys.argv) > 2:
        print("info:传入参数数量错误,请检查运行命令!", VERBOSE, 0)        
        sys.exit(1)
    main(Yaml_Path)