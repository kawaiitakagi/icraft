import sys
sys.path.append(R"../../../Deps/modelzoo")
import icraft
import icraft.xir as ir
import icraft.xrt as rt
import icraft.host_backend as hb
import icraft.buyibackend as bb
from pyrtutils.icraft_utils import *
from pyrtutils.Netinfo import Netinfo
from pyrtutils.utils import *
from pyrtutils.et_device import *
from pyrtutils.calctime_utils import *
import yaml
import os
import numpy as np
import collections


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


if __name__ == "__main__":
    # 获取yaml
    Yaml_Path = "../cfg/bert.yaml"
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

    resRoot = cfg["dataset"]["res"]
    vocab_file = cfg["dataset"]["vocab"]

    
    # 加载指令生成后的网络
    network = ir.Network.CreateFromJsonFile(JSON_PATH)
    network.loadParamsFromFile(RAW_PATH)
    
    netinfo = Netinfo(network)

    # 打开device
    device = openDevice(run_sim, ip, netinfo.mmu or load_mmu)

    # 网络设置
    network_view = network.view(0)  # 选择对网络进行切分

    session = initSession(run_sim, network_view, device, netinfo.mmu or load_mmu, load_speedmode, load_compressFtmp)

    session.enableTimeProfile(True)  #开启计时功能
    session.apply()

    # 输入
    embedding_out = np.fromfile("../io/input/embedding_out.ftmp", dtype=np.float32).reshape(1,128,768)
    input_ids = np.fromfile("../io/input/input_ids.ftmp", dtype=np.float32).reshape(1,128)
    attention_mask =  np.fromfile("../io/input/attention_mask.ftmp", dtype=np.float32).reshape(1,128)
    embedding_out_tensor = rt.Tensor(embedding_out, ir.Layout("**C"))
    input_ids_tensor = rt.Tensor(input_ids, ir.Layout("*C"))
    attention_mask_tensor = rt.Tensor(attention_mask, ir.Layout("*C"))

    # 特征提取阶段socket推理
    output_tensors = session.forward([embedding_out_tensor, input_ids_tensor,attention_mask_tensor])
    if not run_sim: 
        device.reset(1)
        calctime_detail(session,network, name="./"+network.name+"_time.xlsx")

    # 获取结果
    reslist = []
    for tensor in output_tensors:
        reslist.append(np.asarray(tensor))

    # 获取输入的的[MASK]位置下标
    mask_token_index = 4
    # 获取[MASK]位置对应的预测结果
    masked_token_logits = reslist[0][0][mask_token_index]
    # 获取预测结果中概率最高的token索引
    predicted_token_index = np.argmax(masked_token_logits)

    # 输入文本
    text = "I want to [MASK] a new car."
    # 根据字典将预测id映射为文本
    vocab = load_vocab(vocab_file)
    ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in vocab.items()])
    predicted_token = ids_to_tokens[predicted_token_index]
    # 预测的单词替换mask
    completed_text = text.replace('[MASK]', predicted_token)
    # 打印补全后的文本
    if(show):
        print("Tnput text:", text)
        print("Completed text:", completed_text)
    if(save):
        file_name = resRoot + "/result.txt"
        print("Save text in: ", file_name)
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(completed_text)