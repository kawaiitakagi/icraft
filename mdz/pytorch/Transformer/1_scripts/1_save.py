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
    # out = self.postion_embedding(out)
    for encoder in self.encoders:
        out = encoder(out)
    out = out.view(out.size(0), -1)
    # out = torch.mean(out, 1)
    out = self.fc1(out)
    return out


# 参数设置
TRACE_PATH = R'../2_compile/fmodel/Transformer_traced.pt'

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
    def __init__(self, model_name='Transformer', dataset='../0_Chinese-Text-Classification-Pytorch/THUCNews/', embedding='embedding_SougouNews.npz', use_word=False):
        if use_word:
            self.tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
        else:
            self.tokenizer = lambda x: [y for y in x]  # char-level
        self.x = import_module('models.' + model_name)
        self.config = self.x.Config(dataset, embedding)
        self.vocab = pkl.load(open(self.config.vocab_path, 'rb'))
        self.pad_size = self.config.pad_size
        self.model = self.x.Model(self.config).to('cpu')
        self.model.load_state_dict(torch.load(weight_path, map_location='cpu'))

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
            postion_embedding_out = self.model.postion_embedding(embedding_out)

            postion_embedding_out.detach().numpy().astype(np.float32).tofile("../2_compile/qtset/Transformer/postion_embedding_out.ftmp")

            # 保存trace模型
            trace_model = torch.jit.trace(self.model, postion_embedding_out)
            torch.jit.save(trace_model,TRACE_PATH)
            print('TorchScript export success, saved in %s' % TRACE_PATH)


if __name__ == "__main__":
    weight_path = R"../weights/Transformer.ckpt"
    pred = Predict('Transformer')
    # 预测一条
    query = "昨天游戏通关了吗？"
    pred.predict(query)
