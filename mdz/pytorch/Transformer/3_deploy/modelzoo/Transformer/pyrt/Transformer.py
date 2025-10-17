# coding: UTF-8
import sys
sys.path.append(R'../../../../0_Chinese-Text-Classification-Pytorch/')
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


import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif


parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('config', type=str, default='../cfg/TextCNN.yaml', help='path to the config file')
parser.add_argument('--model', type=str, default='Transformer', help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()


def test(config, model, test_iter, resRoot):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'

    with open('./Transformer_metrics.log', 'w') as f:
        f.write(f"Acc: {test_acc:>6.2%}")

    os.makedirs(resRoot, exist_ok=True)
    with open(resRoot + '/Transformer_result.log', 'w') as f:
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
        idx = 0
        for texts, labels in data_iter:
            embedding_out = model.embedding(texts[0])
            postion_embedding_out = model.postion_embedding(embedding_out)
            postion_embedding_out = postion_embedding_out.detach().numpy().astype(np.float32)

            embedding_out_tensor = rt.Tensor(postion_embedding_out, ir.Layout("**C"))
            output_tensors = session.forward([embedding_out_tensor])
            if not run_sim: 
                device.reset(1)
                # calctime_detail(session,network, name="./"+network.name+"_time.xlsx")

            # 获取结果
            reslist = []
            for tensor in output_tensors:
                reslist.append(np.asarray(tensor))
            outputs = torch.tensor(reslist[0])

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
    dataset = '../vocab'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    config.test_path = '../vocab/data/test.txt'
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    
    # 替换 model 的 forward 方法
    # model.forward = new_forward.__get__(model)
    # model.eval()

    # 从yaml里读入配置
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)   
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



    test(config, model, test_iter, resRoot)
    # calctime_detail(session,network, name="./"+network.name+"_time.xlsx")