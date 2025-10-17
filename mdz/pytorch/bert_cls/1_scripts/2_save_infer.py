#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# !/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append(R'../0_Bert-Chinese-Text-Classification-Pytorch/')

import time
import torch
import numpy as np
from train_eval import train, init_network, test
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif
from transformers import BertForSequenceClassification, BertTokenizer, BertModel
from typing import List, Optional, Tuple, Union


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 1                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = '../weights/bert_pretrain'
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
        embedding_out = model.bert.bert.embeddings(input_ids)
        model = torch.jit.load("../2_compile/fmodel/bert_traced.pt")
        outputs = model(embedding_out, input_ids, attention_mask)
        print(outputs)
        predicted_class = torch.argmax(outputs[0], dim=1).item()
    
    return predicted_class

if __name__ == '__main__':
    dataset = '../0_Bert-Chinese-Text-Classification-Pytorch/THUCNews/'  # 数据集

    model_name = 'bert'  # bert
    x = import_module('models.' + model_name)
    config = Config(dataset)


    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    # train
    model = x.Model(config).to(config.device)
    
    # 加载预训练模型权重
    infer_path = '../weights/bert.ckpt'
    model.load_state_dict(torch.load(infer_path))
    
    # 单张推理示例
    text = "一起去学习啊"
    predicted_class = single_inference(config, model, text)
    print(f"Predicted class: {key[predicted_class]}")