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





def BertModel_forward(
    self,
    embedding_output: Optional[torch.Tensor] = None,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
):
    r"""
    encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
        Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
        the model is configured as a decoder.
    encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
        the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

        - 1 for tokens that are **not masked**,
        - 0 for tokens that are **masked**.
    past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
        Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

        If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
        don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
        `decoder_input_ids` of shape `(batch_size, sequence_length)`.
    use_cache (`bool`, *optional*):
        If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
        `past_key_values`).
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if self.config.is_decoder:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
    else:
        use_cache = False

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        input_shape = input_ids.size()
        self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    batch_size, seq_length = input_shape
    device = input_ids.device if input_ids is not None else inputs_embeds.device

    # past_key_values_length
    past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

    if attention_mask is None:
        attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

    if token_type_ids is None:
        if hasattr(self.embeddings, "token_type_ids"):
            buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
            token_type_ids = buffered_token_type_ids_expanded
        else:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

    # If a 2D or 3D attention mask is provided for the cross-attention
    # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
    if self.config.is_decoder and encoder_hidden_states is not None:
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
        encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
    else:
        encoder_extended_attention_mask = None

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
    # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

    # embedding_output = self.embeddings(
    #     input_ids=input_ids,
    #     position_ids=position_ids,
    #     token_type_ids=token_type_ids,
    #     inputs_embeds=inputs_embeds,
    #     past_key_values_length=past_key_values_length,
    # )
    encoder_outputs = self.encoder(
        embedding_output,
        attention_mask=extended_attention_mask,
        head_mask=head_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_extended_attention_mask,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    sequence_output = encoder_outputs[0]
    pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

    # 原返回
    if not return_dict:
        return (sequence_output, pooled_output) + encoder_outputs[1:]



def BertForSequenceClassification_forward(
    self,
    embedding_output: Optional[torch.Tensor] = None,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) :
    r"""
    labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
        config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
        `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.bert(
        embedding_output,
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    pooled_output = outputs[1]

    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)

    loss = None
    if not return_dict:
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output



def model_forward(self, embedding_output, context, mask):
    outputs = self.bert(embedding_output, input_ids=context, attention_mask=mask)
    return outputs



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
        BertModel.forward = BertModel_forward
        BertForSequenceClassification.forward = BertForSequenceClassification_forward
        model.forward = model_forward.__get__(model)
        embedding_out = model.bert.bert.embeddings(input_ids)

        # 保存embedding模型
        TRACE_PATH0 = '../3_deploy/modelzoo/bert_cls/vocab/saved_dict/embedding.onnx'
        torch.onnx.export(
            model.bert.bert.embeddings,  # 模型部分
            input_ids,  # 输入
            TRACE_PATH0,  # 导出路径
            verbose=True,  # 打印详细信息
            opset_version=11,  # ONNX opset版本
            input_names=['input_ids'],  # 输入名称
            output_names=['embedding_output']  # 输出名称
        )
        print('TorchScript export embedding_model success, saved in %s' % TRACE_PATH0)

        outputs = model(embedding_out, input_ids, attention_mask)

        # 保存qtset
        # embedding_out.detach().numpy().astype(np.float32).tofile("../2_compile/qtset/bert/embedding_out.ftmp")
        # input_ids.detach().numpy().astype(np.float32).tofile("../2_compile/qtset/bert/input_ids.ftmp")
        # attention_mask.detach().numpy().astype(np.float32).tofile("../2_compile/qtset/bert/attention_mask.ftmp")
        # print('qtset export success, saved in ../2_compile/qtset/bert')

        # 保存trace模型
        TRACE_PATH = '../2_compile/fmodel/bert_traced.pt'
        trace_model = torch.jit.trace(model, (embedding_out, input_ids, attention_mask))
        torch.jit.save(trace_model,TRACE_PATH)
        print('TorchScript export success, saved in %s' % TRACE_PATH)

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