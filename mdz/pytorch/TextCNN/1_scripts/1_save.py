#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# !/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append(R'../0_Chinese-Text-Classification-Pytorch/')

import icraft
import icraft.xir as ir
import icraft.xrt as rt
import icraft.host_backend as hb
import icraft.buyibackend as bb

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
TRACE_PATH = R'../2_compile/fmodel/TextCNN_traced.pt'

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
    def __init__(self, model_name='TextCNN', dataset='../0_Chinese-Text-Classification-Pytorch/THUCNews/', embedding='embedding_SougouNews.npz', use_word=False):
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
            print('TorchScript export success, saved in %s' % TRACE_PATH)
            num = torch.argmax(outputs)
        return key[int(num)]
    



if __name__ == "__main__":
    pred = Predict('TextCNN')
    # 预测一条
    # query = "学费太贵怎么办？"
    # query = "金融怎么样"
    # query = "今天股票涨了吗？"
    query = "昨天游戏通关了吗？"
    # query = "明天打球去啊"
    print(pred.predict(query))

    
    # # 预测一个列表
    # querys = ["学费太贵怎么办？", "金融怎么样"]
    # print(pred.predict_list(querys))