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

import os
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


def test(config, model, test_iter, resRoot):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'

    with open('./bert_cls_metrics.log', 'w') as f:
        f.write(f"Acc: {test_acc:>6.2%}")

    os.makedirs(resRoot, exist_ok=True)
    with open(resRoot + '/bert_cls_result.log', 'w') as f:
        f.write(msg.format(test_loss, test_acc) + '\n')
        
        f.write("Precision, Recall and F1-Score...\n")
        f.write(test_report + '\n')
        
        f.write("Confusion Matrix...\n")
        f.write(str(test_confusion) + '\n')
        
        time_dif = get_time_dif(start_time)
        f.write("Time usage: " + str(time_dif) + '\n')


    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in tqdm(data_iter, desc="Evaluating", ncols=100):
            embedding_out = model.bert.bert.embeddings(texts[0])

            embedding_out = embedding_out.detach().numpy().astype(np.float32)
            input_ids = texts[0].detach().numpy().astype(np.float32)
            attention_mask = texts[2].detach().numpy().astype(np.float32)
            
            embedding_out_tensor = rt.Tensor(embedding_out, ir.Layout("**C"))
            input_ids_tensor = rt.Tensor(input_ids, ir.Layout("*C"))
            attention_mask_tensor = rt.Tensor(attention_mask, ir.Layout("*C"))

            output_tensors = session.forward([embedding_out_tensor,input_ids_tensor,attention_mask_tensor])
            if not run_sim: 
                device.reset(1)

            outputs = torch.tensor(np.asarray(output_tensors[0]))

            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)


if __name__ == '__main__':

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


    # ---------------------------- model -------------------------------------------------

    dataset = vocab_path  # 数据集

    model_name = 'bert'  # bert
    x = import_module('models.' + model_name)
    config = Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)

    test(config, model, test_iter, resRoot)
