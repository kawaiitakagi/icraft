# coding: UTF-8
import sys
sys.path.append(R'../../../../0_Bert-Chinese-Text-Classification-Pytorch/')
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

from tqdm import tqdm
import time
import torch
import numpy as np
from train_eval import train, init_network, test
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif
import torch.nn.functional as F
from sklearn import metrics
from transformers import BertTokenizer
import platform


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.save_path = '../../../../weights/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 1                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = '../../../../weights/bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号
key = {
    0: 'finance',
    1: 'realty',
    2: 'stocks',
    3: 'education',
    4: 'science',
    5: 'society',
    6: 'politics',
    7: 'sports',
    8: 'game',
    9: 'entertainment'
}


def tokenize_text(text, config, pad_size=32):
    """将单个文本样本转换为模型输入格式"""
    token = config.tokenizer.tokenize(text)
    token = [CLS] + token
    seq_len = len(token)
    mask = []
    token_ids = config.tokenizer.convert_tokens_to_ids(token)

    if pad_size:
        if len(token) < pad_size:
            mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
            token_ids += ([0] * (pad_size - len(token)))
        else:
            mask = [1] * pad_size
            token_ids = token_ids[:pad_size]
            seq_len = pad_size

    return {
        'input_ids': token_ids,
        'attention_mask': mask,
        'seq_len': seq_len
    }


def single_inference(config, model, text):
    # 假设 text 是一个字符串
    tokenized_text = tokenize_text(text, config)
    
    # 将 tokenized_text 转换为模型输入格式
    input_ids = torch.tensor([tokenized_text['input_ids']]).to(config.device)
    attention_mask = torch.tensor([tokenized_text['attention_mask']]).to(config.device)

    # 模型推理
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        embedding_out = model(input_ids)

        embedding_out = embedding_out.detach().numpy().astype(np.float32)
        input_ids = input_ids.detach().numpy().astype(np.float32)
        attention_mask = attention_mask.detach().numpy().astype(np.float32)

        embedding_out_tensor = rt.Tensor(embedding_out, ir.Layout("**C"))
        input_ids_tensor = rt.Tensor(input_ids, ir.Layout("*C"))
        attention_mask_tensor = rt.Tensor(attention_mask, ir.Layout("*C"))

        output_tensors = session.forward([embedding_out_tensor,input_ids_tensor,attention_mask_tensor])
        if not run_sim: 
            device.reset(1)
            calctime_detail(session,network, name="./"+network.name+"_time.xlsx")


        # 获取结果
        reslist = []
        for tensor in output_tensors:
            reslist.append(np.asarray(tensor))
        outputs = torch.tensor(reslist[0])
        # print(outputs)
        predicted_class = torch.argmax(outputs, dim=1).item()
    
    return predicted_class


if __name__ == "__main__":

    # icraft参数
    # 获取yaml
    Yaml_Path = "../cfg/bert_cls.yaml"
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
    vocab_path = cfg["dataset"]["vocab"]


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


    # ----------------------   model  --------------------------
    dataset = '../vocab/'  # 数据集

    model_name = 'bert'  # bert
    x = import_module('models.' + model_name)
    config = Config(dataset)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    
    # 加载预训练模型权重
    model = torch.jit.load("embedding.pt")
    
    # 单张推理示例
    text = "一起去学习啊"
    predicted_class = single_inference(config, model, text)
    print(f"Text Class: {key[predicted_class]}")

    with open(resRoot + '/infer_bert_result.log', 'w') as f:
        f.write(f"Text Class: {key[predicted_class]}")    
    # 互动
    # while True:
    #     text = input("Input Text: ")
    #     if text == "结束":
    #         break
    #     predicted_class = single_inference(config, model, text)
    #     print(f"Text Class: {key[predicted_class]}")
