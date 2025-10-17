#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# !/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append(R'../0_Chinese-Text-Classification-Pytorch/')

import torch
import torch.nn as nn
import pickle as pkl
import numpy as np
from importlib import import_module


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
        self.model.load_state_dict(torch.load(WEIGHT_PATH, map_location='cpu'))

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

    
    def predict_trace(self, query):
        query = [query]
        # 返回预测的索引
        data = self.build_predict_text(query)
        with torch.no_grad():
            self.model.eval()
            embedding_out = self.model.embedding(data[0])
            postion_embedding_out = self.model.postion_embedding(embedding_out)
            trace_model = torch.jit.load(TRACE_PATH)
            outputs_trace = trace_model(postion_embedding_out)

            print("Predict data: " ,outputs_trace.numpy()[0])
            num_trace = torch.argmax(outputs_trace)
        return key[int(num_trace)]



if __name__ == "__main__":
    # 参数设置
    WEIGHT_PATH = R"../weights/Transformer.ckpt"
    TRACE_PATH = R'../2_compile/fmodel/Transformer_traced.pt'
    pred = Predict('Transformer')

    # 预测一条
    query = "昨天游戏通关了吗？"
    print("Input text：  昨天游戏通关了吗？")
    print("Predict Class: " + pred.predict_trace(query))
