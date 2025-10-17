#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# !/usr/bin/env python
# -*- coding: UTF-8 -*-

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



import torch
import torch.nn as nn
import pickle as pkl
import numpy as np
from importlib import import_module


def new_forward(self, out):
    # out = self.embedding(x[0])
    out = out.unsqueeze(1)
    out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
    out = self.dropout(out)
    out = self.fc(out)
    return out


# 参数设置
TRACE_PATH = R'TextCNN_traced.pt'

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


class Predict:
    def __init__(self, model_name='TextCNN', dataset='../io/', embedding='embedding_SougouNews.npz', use_word=False):
        if use_word:
            self.tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
        else:
            self.tokenizer = lambda x: [y for y in x]  # char-level
        self.x = import_module('models.' + model_name)
        self.config = self.x.Config(dataset, embedding)
        self.vocab = pkl.load(open(self.config.vocab_path, 'rb'))
        self.pad_size = self.config.pad_size
        self.model = self.x.Model(self.config).to('cpu')
        self.model.load_state_dict(torch.load(self.config.save_path, map_location='cpu'))

    def build_predict_text(self, texts):
        words_lines = []
        seq_lens = []
        for text in texts:
            words_line = []
            token = self.tokenizer(text)
            seq_len = len(token)
            if self.pad_size:
                if len(token) < self.pad_size:
                    token.extend(['<PAD>'] * (self.pad_size - len(token)))
                else:
                    token = token[:self.pad_size]
                    seq_len = self.pad_size
            # word to id
            for word in token:
                words_line.append(self.vocab.get(word, self.vocab.get('<UNK>')))
            words_lines.append(words_line)
            seq_lens.append(seq_len)

        return torch.LongTensor(words_lines), torch.LongTensor(seq_lens)

    def predict(self, query):
        print("Base infer result: ")
        query = [query]
        # 返回预测的索引
        data = self.build_predict_text(query)
        with torch.no_grad():
            # 替换 model 的 forward 方法
            self.model.forward = new_forward.__get__(self.model)
            self.model.eval()

            embedding_out = self.model.embedding(data[0])

            outputs = self.model(embedding_out)
            print(outputs)
            # 保存trace模型
            trace_model = torch.jit.trace(self.model, embedding_out)
            torch.jit.save(trace_model,TRACE_PATH)

            num = torch.argmax(outputs)
        return key[int(num)]
    
    def predict_trace(self, query):
        print("Trace infer result: ")
        query = [query]
        # 返回预测的索引
        data = self.build_predict_text(query)
        with torch.no_grad():
            embedding_out = self.model.embedding(data[0])
            trace_model = torch.jit.load(TRACE_PATH)
            outputs_trace = trace_model(embedding_out)
            print(outputs_trace)
            num_trace = torch.argmax(outputs_trace)
        return key[int(num_trace)]

    def predict_icraft(self, query):
        # print("Input text：", query)
        query = [query]
        # 返回预测的索引
        data = self.build_predict_text(query)
        with torch.no_grad():
            embedding_out = self.model.embedding(data[0])
            embedding_out = embedding_out.detach().numpy().astype(np.float32)
            # embedding_out.tofile(R"../2_compile/qtset/TextCNN/embedding_out.ftmp")
            embedding_out_tensor = rt.Tensor(embedding_out, ir.Layout("**C"))        
            # 特征提取阶段socket推理
            output_tensors = session.forward([embedding_out_tensor])
            if not run_sim: 
                device.reset(1)
            # 获取结果
            reslist = []
            for tensor in output_tensors:
                reslist.append(np.asarray(tensor))
            outputs_icraft = torch.tensor(reslist[0])

            num_icraft = torch.argmax(outputs_icraft)
        return key[int(num_icraft)]


if __name__ == "__main__":
    pred = Predict('TextCNN')
    # 预测一条
    query = "昨天游戏通关了吗？"

    # 获取yaml
    Yaml_Path = "../cfg/TextCNN.yaml"
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
    
    result = pred.predict_icraft(query)

    with open(resRoot + '/infer_TextCNN_result.log', 'w', encoding='utf-8') as f:
        f.write("Input text: " )
        f.write(query+ '\n' )
        f.write("icraft_infer class:")
        f.write(result + '\n')

    calctime_detail(session,network, name="./"+network.name+"_time.xlsx")