import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import time
from collections import OrderedDict
import types
from functools import partial
import math
import numpy as np

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from pytracking.evaluation import get_dataset
from pytracking.evaluation.running import _save_tracker_output
from pytracking.evaluation import Tracker

from tracking.basic_model.et_tracker import ET_Tracker
from tracking.basic_model.exemplar_transformer import ExemplarTransformer, AveragePooler, SqueezeExcite, _pair
from tracking.basic_model.exemplar_transformer import resolve_se_args,_get_activation_fn,get_initializer
from pytracking.utils import TrackerParams
from pytracking.tracker.et_tracker.et_tracker import TransconverTracker

from lib.models.super_model_DP import Super_model_DP
from lib.models.model_parts import *
import lib.models.models as lighttrack_model
from lib.utils.utils import load_lighttrack_model

# ICRAFT NOTE:
# 为了消除floor_divide算子，需要构建4个不同参数的ExemplarTransformer
class ExemplarTransformer256_5(nn.Module):

    def __init__(self, in_channels=256, 
                       out_channels=256, 
                       dw_padding=2, 
                       pw_padding=0,
                       dw_stride=1, 
                       pw_stride=1,
                       e_exemplars=4, 
                       temperature=2, 
                       hidden_dim=256, 
                       dw_kernel_size=5, 
                       pw_kernel_size=1,
                       layer_norm_eps = 1e-05,
                       dim_feedforward = 1024, # 2048,
                       ff_dropout = 0.1,
                       ff_activation = "relu",
                       num_heads = 8,
                       seq_red = 1,
                       se_ratio = 0.5,
                       se_kwargs = None,
                       se_act_layer = "relu",
                       norm_layer = nn.BatchNorm2d,
                       norm_kwargs = None,
                       sm_normalization = True,
                       dropout = False,
                       dropout_rate = 0.1) -> None:
        super(ExemplarTransformer256_5, self).__init__()

        '''
        
        Sub Models:
            - average_pooler: attention module
            - K (keys): Representing the last layer of the average pooler 
                        K is used for the computation of the mixing weights.
                        The mixing weights are used for the both the spatial as well as the
                        pointwise convolution
            - V (values): Representing the different kernels. 
                          There have to be two sets of values, one for the spatial and one for the pointwise 
                          convolution. The shape of the kernels differ. 


        Args:
            - in_channels: number of input channels
            - out_channels: number of output channels
            - padding: input padding for when applying kernel
            - stride: stride for kernel application
            - e_exemplars: number of expert kernels
            - temperature: temperature for softmax
            - hidden_dim: hidden dimension used in the average pooler
            - kernel_size: kernel size used for the weight shape computation
            - layernorm eps: used for layer norm after the convolution operation
            - dim_feedforward: dimension for FF network after attention module,
            - ff_dropout: dropout rate for FF network after attention module
            - activation: activation function for FF network after attention module
            - num_heads: number of heads
            - seq_red: sequence reduction dimension for the global average pooling operation


        '''

        ## general parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.e_exemplars = e_exemplars
        norm_kwargs = norm_kwargs or {}
        self.hidden_dim = hidden_dim
        self.sm_norm = sm_normalization
        self.K = nn.Parameter(torch.randn(e_exemplars, hidden_dim)) # could be an embedding / a mapping from X to K instead of pre-learned
        self.K_T = None
        self.dropout = dropout
        self.do = nn.Dropout(dropout_rate)

        ## average pool 
        self.temperature = temperature
        self.average_pooler = AveragePooler(seq_red=seq_red, c_dim=in_channels, hidden_dim=hidden_dim) #.cuda()
        self.softmax = nn.Softmax(dim=-1)
        
        ## multihead setting
        self.H = num_heads
        self.head_dim = self.hidden_dim // self.H

        ## depthwise convolution parameters
        self.dw_groups = self.out_channels
        self.dw_kernel_size = _pair(dw_kernel_size)
        self.dw_padding = dw_padding
        self.dw_stride = dw_stride
        self.dw_weight_shape = (self.out_channels, self.in_channels // self.dw_groups) + self.dw_kernel_size
        dw_weight_num_param = 1
        for wd in self.dw_weight_shape:
            dw_weight_num_param *= wd
        self.V_dw = nn.Parameter(torch.Tensor(e_exemplars, dw_weight_num_param))
        self.dw_bn = norm_layer(self.in_channels, **norm_kwargs)
        self.dw_act = nn.ReLU(inplace=True)

        ## pointwise convolution parameters
        self.pw_groups = 1
        self.pw_kernel_size = _pair(pw_kernel_size)
        self.pw_padding = pw_padding
        self.pw_stride = pw_stride
        self.pw_weight_shape = (self.out_channels, self.in_channels // self.pw_groups) + self.pw_kernel_size
        pw_weight_num_param = 1
        for wd in self.pw_weight_shape:
            pw_weight_num_param *= wd
        self.V_pw = nn.Parameter(torch.Tensor(e_exemplars, pw_weight_num_param))
        self.pw_bn = norm_layer(self.out_channels, **norm_kwargs)
        self.pw_act = nn.ReLU(inplace=False)

        ## Squeeze-and-excitation
        if se_ratio is not None and se_ratio > 0.:
            se_kwargs = resolve_se_args(se_kwargs, self.in_channels, nn.ReLU) #_get_activation_fn(se_act_layer))
            self.se = SqueezeExcite(self.in_channels, se_ratio=se_ratio, **se_kwargs)
        
        ## Implementation of Feedforward model after the QKV part
        self.linear1 = nn.Linear(self.out_channels, dim_feedforward)
        self.ff_dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, self.out_channels)
        self.norm1 = nn.LayerNorm(self.out_channels, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(self.out_channels, eps=layer_norm_eps)
        self.ff_dropout1 = nn.Dropout(ff_dropout)
        self.ff_dropout2 = nn.Dropout(ff_dropout)
        self.ff_activation = _get_activation_fn(ff_activation)

        # initialize the kernels 
        self.reset_parameters()

    def reset_parameters(self):
        init_weight_dw = get_initializer(
            partial(nn.init.kaiming_uniform_, a=math.sqrt(5)), self.e_exemplars, self.dw_weight_shape)
        init_weight_dw(self.V_dw)

        init_weight_pw = get_initializer(
            partial(nn.init.kaiming_uniform_, a=math.sqrt(5)), self.e_exemplars, self.pw_weight_shape)
        init_weight_pw(self.V_pw)  

    def forward(self, x):

        residual = x
        
        # X: [B,C,H,W]

        # apply average pooler
        q = self.average_pooler(x)
        d_k = q.shape[-1]
        # Q: [B,S,C]

        # ICRAFT NOTE:
        # 计算Keys外积的时候，K参数的转置需要提前算好，运行时不支持。这一步在顶层ET_Tracker.template函数里完成
        # outer product with keys
        #qk = einsum('b n c, k c -> b n k', q, self.K) # K^T: [C, K] QK^T: [B,S,K]
        # qk = torch.matmul(q, self.K.T)
        qk = torch.matmul(q, self.K_T)
        
        # if self.sm_norm:
        qk = 1/math.sqrt(d_k) * qk

        # apply softmax 
        attn = self.softmax(qk/2) # -> [batch_size, e_exemplars]
        
        # multiply attention map with values 
        #dw_qkv_kernel = einsum('b s k, k e -> b s e', attn, self.V_dw) # V: [K, E_dw]
        #pw_qkv_kernel = einsum('b s k, k e -> b s e', attn, self.V_pw) # V: [K, E_pw]
        dw_qkv_kernel = torch.matmul(attn, self.V_dw) # V: [K, E_dw]
        pw_qkv_kernel = torch.matmul(attn, self.V_pw) # V: [K, E_pw]

        ###########################################################################################
        ####### convolve input with the output instead of adding it to it in a residual way #######
        ###########################################################################################

        ## dw conv
        B, C, H, W = x.shape #[1，256，18，18]
        # ICRAFT NOTE:
        # 为了消除floor_divide算子，预先计算好dw_weight_shape
        # dw conv
        # dw_weight_shape = (B * self.out_channels, self.in_channels // self.dw_groups) + self.dw_kernel_size

        # dw_weight = dw_qkv_kernel.view(dw_weight_shape)
        dw_weight = dw_qkv_kernel.view(256,1,5,5)
        # ICRAFT NOTE:
        # 消除无效reshape
        # reshape the input
        # x = x.reshape(1, 256, 18, 18) #(1, B * C, H, W)
        
        # apply convolution
        x = F.conv2d(x, dw_weight, bias=None, stride=1, padding=2, groups=256)
            # x, dw_weight, bias=None, stride=self.dw_stride, padding=self.dw_padding, 
            # groups=self.dw_groups * B)
        
        # x = x.permute([1, 0, 2, 3]).view(B, self.out_channels, x.shape[-2], x.shape[-1])
        x = x.permute([1, 0, 2, 3]).view(1, 256, 18, 18)
        x = self.dw_bn(x)
        x = self.dw_act(x)

        ## SE
        x = self.se(x)

        ## pw conv
        B, C, H, W = x.shape #[1,256,18,18]

        # ICRAFT NOTE:
        # 为了消除floor_divide算子，预先计算好pw_weight_shape
        # dw conv
        # pw_weight_shape = (B * self.out_channels, self.in_channels // self.pw_groups) + self.pw_kernel_size #[256,256,1,1]
        # pw_weight = pw_qkv_kernel.view(pw_weight_shape)
        pw_weight = pw_qkv_kernel.view(256,256,1,1)
        
        # ICRAFT NOTE:
        # 消除无效view算子
        # reshape the input
        # x = x.view(1, 256, 18, 18) #(1, B * C, H, W)
        
        # apply convolution
        x = F.conv2d(x, pw_weight, bias=None, stride=1, padding=0, groups=1)
            # x, pw_weight, bias=None, stride=self.pw_stride, padding=self.pw_padding, 
            # groups=self.pw_groups * B)
        
        # x = x.permute([1, 0, 2, 3]).view(B, self.out_channels, x.shape[-2], x.shape[-1])
        x = x.permute([1, 0, 2, 3]).view(1, 256, 18, 18)
        x = self.pw_bn(x)
        x = self.pw_act(x)


        # if self.dropout:
        #     x = x + self.do(residual)
        # else:
        x = x + residual

        # reshape output of convolution operation
        # out = x.view(B, self.out_channels, -1).permute(0,2,1)
        out = x.view(1, 256, -1).permute(0,2,1)
        
        # FF network 
        out = self.norm1(out) #[1,324,256]
        out2 = self.linear2(self.ff_dropout(self.ff_activation(self.linear1(out))))
        out = out + self.ff_dropout2(out2)
        out = self.norm2(out)
        # out = out.permute(0,2,1).view(B,C,H,W)
        out = out.permute(0,2,1).view(1,256,18,18)
        
        return out
  
class ExemplarTransformer256_3(nn.Module):

    def __init__(self, in_channels=256, 
                       out_channels=256, 
                       dw_padding=1, 
                       pw_padding=0,
                       dw_stride=1, 
                       pw_stride=1,
                       e_exemplars=4, 
                       temperature=2, 
                       hidden_dim=256, 
                       dw_kernel_size=3, 
                       pw_kernel_size=1,
                       layer_norm_eps = 1e-05,
                       dim_feedforward = 1024, # 2048,
                       ff_dropout = 0.1,
                       ff_activation = "relu",
                       num_heads = 8,
                       seq_red = 1,
                       se_ratio = 0.5,
                       se_kwargs = None,
                       se_act_layer = "relu",
                       norm_layer = nn.BatchNorm2d,
                       norm_kwargs = None,
                       sm_normalization = True,
                       dropout = False,
                       dropout_rate = 0.1) -> None:
        super(ExemplarTransformer256_3, self).__init__()

        '''
        
        Sub Models:
            - average_pooler: attention module
            - K (keys): Representing the last layer of the average pooler 
                        K is used for the computation of the mixing weights.
                        The mixing weights are used for the both the spatial as well as the
                        pointwise convolution
            - V (values): Representing the different kernels. 
                          There have to be two sets of values, one for the spatial and one for the pointwise 
                          convolution. The shape of the kernels differ. 


        Args:
            - in_channels: number of input channels
            - out_channels: number of output channels
            - padding: input padding for when applying kernel
            - stride: stride for kernel application
            - e_exemplars: number of expert kernels
            - temperature: temperature for softmax
            - hidden_dim: hidden dimension used in the average pooler
            - kernel_size: kernel size used for the weight shape computation
            - layernorm eps: used for layer norm after the convolution operation
            - dim_feedforward: dimension for FF network after attention module,
            - ff_dropout: dropout rate for FF network after attention module
            - activation: activation function for FF network after attention module
            - num_heads: number of heads
            - seq_red: sequence reduction dimension for the global average pooling operation


        '''

        ## general parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.e_exemplars = e_exemplars
        norm_kwargs = norm_kwargs or {}
        self.hidden_dim = hidden_dim
        self.sm_norm = sm_normalization
        self.K = nn.Parameter(torch.randn(e_exemplars, hidden_dim)) # could be an embedding / a mapping from X to K instead of pre-learned
        self.K_T = None
        self.dropout = dropout
        self.do = nn.Dropout(dropout_rate)

        ## average pool 
        self.temperature = temperature
        self.average_pooler = AveragePooler(seq_red=seq_red, c_dim=in_channels, hidden_dim=hidden_dim) #.cuda()
        self.softmax = nn.Softmax(dim=-1)
        
        ## multihead setting
        self.H = num_heads
        self.head_dim = self.hidden_dim // self.H

        ## depthwise convolution parameters
        self.dw_groups = self.out_channels
        self.dw_kernel_size = _pair(dw_kernel_size)
        self.dw_padding = dw_padding
        self.dw_stride = dw_stride
        self.dw_weight_shape = (self.out_channels, self.in_channels // self.dw_groups) + self.dw_kernel_size
        dw_weight_num_param = 1
        for wd in self.dw_weight_shape:
            dw_weight_num_param *= wd
        self.V_dw = nn.Parameter(torch.Tensor(e_exemplars, dw_weight_num_param))
        self.dw_bn = norm_layer(self.in_channels, **norm_kwargs)
        self.dw_act = nn.ReLU(inplace=True)

        ## pointwise convolution parameters
        self.pw_groups = 1
        self.pw_kernel_size = _pair(pw_kernel_size)
        self.pw_padding = pw_padding
        self.pw_stride = pw_stride
        self.pw_weight_shape = (self.out_channels, self.in_channels // self.pw_groups) + self.pw_kernel_size
        pw_weight_num_param = 1
        for wd in self.pw_weight_shape:
            pw_weight_num_param *= wd
        self.V_pw = nn.Parameter(torch.Tensor(e_exemplars, pw_weight_num_param))
        self.pw_bn = norm_layer(self.out_channels, **norm_kwargs)
        self.pw_act = nn.ReLU(inplace=False)

        ## Squeeze-and-excitation
        if se_ratio is not None and se_ratio > 0.:
            se_kwargs = resolve_se_args(se_kwargs, self.in_channels, nn.ReLU) #_get_activation_fn(se_act_layer))
            self.se = SqueezeExcite(self.in_channels, se_ratio=se_ratio, **se_kwargs)
        
        ## Implementation of Feedforward model after the QKV part
        self.linear1 = nn.Linear(self.out_channels, dim_feedforward)
        self.ff_dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, self.out_channels)
        self.norm1 = nn.LayerNorm(self.out_channels, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(self.out_channels, eps=layer_norm_eps)
        self.ff_dropout1 = nn.Dropout(ff_dropout)
        self.ff_dropout2 = nn.Dropout(ff_dropout)
        self.ff_activation = _get_activation_fn(ff_activation)

        # initialize the kernels 
        self.reset_parameters()

    def reset_parameters(self):
        init_weight_dw = get_initializer(
            partial(nn.init.kaiming_uniform_, a=math.sqrt(5)), self.e_exemplars, self.dw_weight_shape)
        init_weight_dw(self.V_dw)

        init_weight_pw = get_initializer(
            partial(nn.init.kaiming_uniform_, a=math.sqrt(5)), self.e_exemplars, self.pw_weight_shape)
        init_weight_pw(self.V_pw)
    
    def forward(self, x):

        residual = x
        
        # X: [B,C,H,W]

        # apply average pooler
        q = self.average_pooler(x)
        d_k = q.shape[-1]
        # Q: [B,S,C]

        # ICRAFT NOTE:
        # 计算Keys外积的时候，K参数的转置需要提前算好，运行时不支持。这一步在顶层ET_Tracker.template函数里完成
        # outer product with keys
        #qk = einsum('b n c, k c -> b n k', q, self.K) # K^T: [C, K] QK^T: [B,S,K]
        # qk = torch.matmul(q, self.K.T)
        qk = torch.matmul(q, self.K_T)
        
        # if self.sm_norm:
        qk = 1/math.sqrt(d_k) * qk

        # apply softmax 
        attn = self.softmax(qk/2) #self.temperature) # -> [batch_size, e_exemplars]
        
        # multiply attention map with values 
        #dw_qkv_kernel = einsum('b s k, k e -> b s e', attn, self.V_dw) # V: [K, E_dw]
        #pw_qkv_kernel = einsum('b s k, k e -> b s e', attn, self.V_pw) # V: [K, E_pw]
        dw_qkv_kernel = torch.matmul(attn, self.V_dw) # V: [K, E_dw]
        pw_qkv_kernel = torch.matmul(attn, self.V_pw) # V: [K, E_pw]

        ###########################################################################################
        ####### convolve input with the output instead of adding it to it in a residual way #######
        ###########################################################################################

        ## dw conv
        B, C, H, W = x.shape

        # ICRAFT NOTE:
        # 为了消除floor_divide算子，预先计算好dw_weight_shape
        # dw conv
        # dw_weight_shape = (B * self.out_channels, self.in_channels // self.dw_groups) + self.dw_kernel_size
        dw_weight = dw_qkv_kernel.view(256,1,3,3)
        
        # reshape the input
        # x = x.reshape(1,256,18,18) #(1, B * C, H, W)
        # ICRAFT NOTE:
        # 消除无效reshape
        # apply convolution
        x = F.conv2d(x, dw_weight, bias=None, stride=1, padding=1, groups=256)
            # x, dw_weight, bias=None, stride=self.dw_stride, padding=self.dw_padding, 
            # groups=self.dw_groups * B)
        
        x = x.permute([1, 0, 2, 3]).view(1,256,18,18) #(B, self.out_channels, x.shape[-2], x.shape[-1])
        x = self.dw_bn(x)
        x = self.dw_act(x)

        ## SE
        x = self.se(x)

        ## pw conv
        B, C, H, W = x.shape
        # ICRAFT NOTE:
        # 为了消除floor_divide算子，预先计算好pw_weight_shape
        # dw conv
        # pw_weight_shape = (B * self.out_channels, self.in_channels // self.pw_groups) + self.pw_kernel_size
        pw_weight = pw_qkv_kernel.view(256,256,1,1)
        # ICRAFT NOTE:
        # 消除无效view算子
        # reshape the input
        # x = x.view(1,256,18,18)#(1, B * C, H, W)
        
        # apply convolution
        x = F.conv2d(x, pw_weight, bias=None, stride=1, padding=0, groups=1)
            # x, pw_weight, bias=None, stride=self.pw_stride, padding=self.pw_padding, 
            # groups=self.pw_groups * B)
        
        x = x.permute([1, 0, 2, 3]).view(1,256,18,18) #(B, self.out_channels, x.shape[-2], x.shape[-1])
        x = self.pw_bn(x)
        x = self.pw_act(x)

        # if self.dropout:
        #     x = x + self.do(residual)
        # else:
        x = x + residual

        
        # reshape output of convolution operation
        # out = x.view(B, self.out_channels, -1).permute(0,2,1)
        out = x.view(1, 256, -1).permute(0,2,1)
        
        # FF network 
        out = self.norm1(out)
        out2 = self.linear2(self.ff_dropout(self.ff_activation(self.linear1(out))))
        out = out + self.ff_dropout2(out2)
        out = self.norm2(out)
        out = out.permute(0,2,1).view(1,256,18,18) #(B,C,H,W)
        
        return out
  
class ExemplarTransformer192_3(nn.Module):

    def __init__(self, in_channels=192, 
                       out_channels=192, 
                       dw_padding = 1, 
                       pw_padding=0,
                       dw_stride=1, 
                       pw_stride=1,
                       e_exemplars=4, 
                       temperature=2, 
                       hidden_dim=256, 
                       dw_kernel_size=3, 
                       pw_kernel_size=1,
                       layer_norm_eps = 1e-05,
                       dim_feedforward = 1024, # 2048,
                       ff_dropout = 0.1,
                       ff_activation = "relu",
                       num_heads = 8,
                       seq_red = 1,
                       se_ratio = 0.5,
                       se_kwargs = None,
                       se_act_layer = "relu",
                       norm_layer = nn.BatchNorm2d,
                       norm_kwargs = None,
                       sm_normalization = True,
                       dropout = False,
                       dropout_rate = 0.1) -> None:
        super(ExemplarTransformer192_3, self).__init__()

        '''
        
        Sub Models:
            - average_pooler: attention module
            - K (keys): Representing the last layer of the average pooler 
                        K is used for the computation of the mixing weights.
                        The mixing weights are used for the both the spatial as well as the
                        pointwise convolution
            - V (values): Representing the different kernels. 
                          There have to be two sets of values, one for the spatial and one for the pointwise 
                          convolution. The shape of the kernels differ. 


        Args:
            - in_channels: number of input channels
            - out_channels: number of output channels
            - padding: input padding for when applying kernel
            - stride: stride for kernel application
            - e_exemplars: number of expert kernels
            - temperature: temperature for softmax
            - hidden_dim: hidden dimension used in the average pooler
            - kernel_size: kernel size used for the weight shape computation
            - layernorm eps: used for layer norm after the convolution operation
            - dim_feedforward: dimension for FF network after attention module,
            - ff_dropout: dropout rate for FF network after attention module
            - activation: activation function for FF network after attention module
            - num_heads: number of heads
            - seq_red: sequence reduction dimension for the global average pooling operation


        '''

        ## general parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.e_exemplars = e_exemplars
        norm_kwargs = norm_kwargs or {}
        self.hidden_dim = hidden_dim
        self.sm_norm = sm_normalization
        self.K = nn.Parameter(torch.randn(e_exemplars, hidden_dim)) # could be an embedding / a mapping from X to K instead of pre-learned
        self.K_T = None
        self.dropout = dropout
        self.do = nn.Dropout(dropout_rate)

        ## average pool 
        self.temperature = temperature
        self.average_pooler = AveragePooler(seq_red=seq_red, c_dim=in_channels, hidden_dim=hidden_dim) #.cuda()
        self.softmax = nn.Softmax(dim=-1)
        
        ## multihead setting
        self.H = num_heads
        self.head_dim = self.hidden_dim // self.H

        ## depthwise convolution parameters
        self.dw_groups = self.out_channels
        self.dw_kernel_size = _pair(dw_kernel_size)
        self.dw_padding = dw_padding
        self.dw_stride = dw_stride
        self.dw_weight_shape = (self.out_channels, self.in_channels // self.dw_groups) + self.dw_kernel_size
        dw_weight_num_param = 1
        for wd in self.dw_weight_shape:
            dw_weight_num_param *= wd
        self.V_dw = nn.Parameter(torch.Tensor(e_exemplars, dw_weight_num_param))
        self.dw_bn = norm_layer(self.in_channels, **norm_kwargs)
        self.dw_act = nn.ReLU(inplace=True)

        ## pointwise convolution parameters
        self.pw_groups = 1
        self.pw_kernel_size = _pair(pw_kernel_size)
        self.pw_padding = pw_padding
        self.pw_stride = pw_stride
        self.pw_weight_shape = (self.out_channels, self.in_channels // self.pw_groups) + self.pw_kernel_size
        pw_weight_num_param = 1
        for wd in self.pw_weight_shape:
            pw_weight_num_param *= wd
        self.V_pw = nn.Parameter(torch.Tensor(e_exemplars, pw_weight_num_param))
        self.pw_bn = norm_layer(self.out_channels, **norm_kwargs)
        self.pw_act = nn.ReLU(inplace=False)

        ## Squeeze-and-excitation
        if se_ratio is not None and se_ratio > 0.:
            se_kwargs = resolve_se_args(se_kwargs, self.in_channels, nn.ReLU) #_get_activation_fn(se_act_layer))
            self.se = SqueezeExcite(self.in_channels, se_ratio=se_ratio, **se_kwargs)
        
        ## Implementation of Feedforward model after the QKV part
        self.linear1 = nn.Linear(self.out_channels, dim_feedforward)
        self.ff_dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, self.out_channels)
        self.norm1 = nn.LayerNorm(self.out_channels, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(self.out_channels, eps=layer_norm_eps)
        self.ff_dropout1 = nn.Dropout(ff_dropout)
        self.ff_dropout2 = nn.Dropout(ff_dropout)
        self.ff_activation = _get_activation_fn(ff_activation)

        # initialize the kernels 
        self.reset_parameters()

    def reset_parameters(self):
        init_weight_dw = get_initializer(
            partial(nn.init.kaiming_uniform_, a=math.sqrt(5)), self.e_exemplars, self.dw_weight_shape)
        init_weight_dw(self.V_dw)

        init_weight_pw = get_initializer(
            partial(nn.init.kaiming_uniform_, a=math.sqrt(5)), self.e_exemplars, self.pw_weight_shape)
        init_weight_pw(self.V_pw)
    
    def forward(self, x):
        residual = x
        # X: [B,C,H,W]
        # apply average pooler
        q = self.average_pooler(x) #(1,1,256)
        # d_k = q.shape[-1]
        # Q: [B,S,C]
        # ICRAFT NOTE:
        # 计算Keys外积的时候，K参数的转置需要提前算好，运行时不支持。这一步在顶层ET_Tracker.template函数里完成
        # outer product with keys
        #qk = einsum('b n c, k c -> b n k', q, self.K) # K^T: [C, K] QK^T: [B,S,K]
        # qk = torch.matmul(q, self.K.T)
        qk = torch.matmul(q, self.K_T)
        
        # if self.sm_norm:
        qk = 1/16.0 * qk #qk = 1/math.sqrt(d_k) * qk

        # apply softmax 
        attn = self.softmax(qk/2) # -> [batch_size, e_exemplars]
        
        # multiply attention map with values 
        #dw_qkv_kernel = einsum('b s k, k e -> b s e', attn, self.V_dw) # V: [K, E_dw]
        #pw_qkv_kernel = einsum('b s k, k e -> b s e', attn, self.V_pw) # V: [K, E_pw]
        dw_qkv_kernel = torch.matmul(attn, self.V_dw) # V: [K, E_dw]
        pw_qkv_kernel = torch.matmul(attn, self.V_pw) # V: [K, E_pw]

        ###########################################################################################
        ####### convolve input with the output instead of adding it to it in a residual way #######
        ###########################################################################################

        ## dw conv
        B, C, H, W = x.shape
        # ICRAFT NOTE:
        # 为了消除floor_divide算子，预先计算好dw_weight_shape
        # dw conv
        # dw_weight_shape = (B * self.out_channels, self.in_channels // self.dw_groups) + self.dw_kernel_size
        dw_weight = dw_qkv_kernel.view(192,1,3,3)
        # ICRAFT NOTE:
        # 消除无效reshape
        # reshape the input
        # x = x.reshape(1,192,18,18) #(1, B * C, H, W)
        
        # apply convolution
        x = F.conv2d(x, dw_weight, bias=None, stride=1, padding=1, groups=192)
            # x, dw_weight, bias=None, stride=self.dw_stride, padding=self.dw_padding, 
            # groups=self.dw_groups * B)
        
        x = x.permute([1, 0, 2, 3]).view(1,192,18,18) #(B, self.out_channels, x.shape[-2], x.shape[-1])
        x = self.dw_bn(x)
        x = self.dw_act(x)

        ## SE
        x = self.se(x)

        ## pw conv
        B, C, H, W = x.shape
        # ICRAFT NOTE:
        # 为了消除floor_divide算子，预先计算好pw_weight_shape
        # dw conv
        # pw_weight_shape = (B * self.out_channels, self.in_channels // self.pw_groups) + self.pw_kernel_size
        pw_weight = pw_qkv_kernel.view(192,192,1,1)
        # ICRAFT NOTE:
        # 消除无效view算子
        # reshape the input
        # x = x.view(1,192,18,18) #(1, B * C, H, W)
        
        # apply convolution
        x = F.conv2d(x, pw_weight, bias=None, stride=1, padding=0, groups=1)
            # x, pw_weight, bias=None, stride=self.pw_stride, padding=self.pw_padding, 
            # groups=self.pw_groups * B)
        
        x = x.permute([1, 0, 2, 3]).view(1,192,18,18) #(B, self.out_channels, x.shape[-2], x.shape[-1])
        x = self.pw_bn(x)
        x = self.pw_act(x)


        # if self.dropout:
        #     x = x + self.do(residual)
        # else:
        x = x + residual

        # reshape output of convolution operation
        # out = x.view(B, self.out_channels, -1).permute(0,2,1)
        out = x.view(1, 192, -1).permute(0,2,1)
        
        # FF network 
        out = self.norm1(out)
        out2 = self.linear2(self.ff_dropout(self.ff_activation(self.linear1(out))))
        out = out + self.ff_dropout2(out2)
        out = self.norm2(out)
        out = out.permute(0,2,1).view(1,192,18,18) #(B,C,H,W)
        
        return out
  
class ExemplarTransformer192_5(nn.Module):

    def __init__(self, in_channels=192, 
                       out_channels=192, 
                       dw_padding=2, 
                       pw_padding=0,
                       dw_stride=1, 
                       pw_stride=1,
                       e_exemplars=4, 
                       temperature=2, 
                       hidden_dim=256, 
                       dw_kernel_size=5, 
                       pw_kernel_size=1,
                       layer_norm_eps = 1e-05,
                       dim_feedforward = 1024, # 2048,
                       ff_dropout = 0.1,
                       ff_activation = "relu",
                       num_heads = 8,
                       seq_red = 1,
                       se_ratio = 0.5,
                       se_kwargs = None,
                       se_act_layer = "relu",
                       norm_layer = nn.BatchNorm2d,
                       norm_kwargs = None,
                       sm_normalization = True,
                       dropout = False,
                       dropout_rate = 0.1) -> None:
        super(ExemplarTransformer192_5, self).__init__()

        '''
        
        Sub Models:
            - average_pooler: attention module
            - K (keys): Representing the last layer of the average pooler 
                        K is used for the computation of the mixing weights.
                        The mixing weights are used for the both the spatial as well as the
                        pointwise convolution
            - V (values): Representing the different kernels. 
                          There have to be two sets of values, one for the spatial and one for the pointwise 
                          convolution. The shape of the kernels differ. 


        Args:
            - in_channels: number of input channels
            - out_channels: number of output channels
            - padding: input padding for when applying kernel
            - stride: stride for kernel application
            - e_exemplars: number of expert kernels
            - temperature: temperature for softmax
            - hidden_dim: hidden dimension used in the average pooler
            - kernel_size: kernel size used for the weight shape computation
            - layernorm eps: used for layer norm after the convolution operation
            - dim_feedforward: dimension for FF network after attention module,
            - ff_dropout: dropout rate for FF network after attention module
            - activation: activation function for FF network after attention module
            - num_heads: number of heads
            - seq_red: sequence reduction dimension for the global average pooling operation


        '''

        ## general parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.e_exemplars = e_exemplars
        norm_kwargs = norm_kwargs or {}
        self.hidden_dim = hidden_dim
        self.sm_norm = sm_normalization
        self.K = nn.Parameter(torch.randn(e_exemplars, hidden_dim)) # could be an embedding / a mapping from X to K instead of pre-learned
        self.K_T = None
        self.dropout = dropout
        self.do = nn.Dropout(dropout_rate)

        ## average pool 
        self.temperature = temperature
        self.average_pooler = AveragePooler(seq_red=seq_red, c_dim=in_channels, hidden_dim=hidden_dim) #.cuda()
        self.softmax = nn.Softmax(dim=-1)
        
        ## multihead setting
        self.H = num_heads
        self.head_dim = self.hidden_dim // self.H

        ## depthwise convolution parameters
        self.dw_groups = self.out_channels
        self.dw_kernel_size = _pair(dw_kernel_size)
        self.dw_padding = dw_padding
        self.dw_stride = dw_stride
        self.dw_weight_shape = (self.out_channels, self.in_channels // self.dw_groups) + self.dw_kernel_size
        dw_weight_num_param = 1
        for wd in self.dw_weight_shape:
            dw_weight_num_param *= wd
        self.V_dw = nn.Parameter(torch.Tensor(e_exemplars, dw_weight_num_param))
        self.dw_bn = norm_layer(self.in_channels, **norm_kwargs)
        self.dw_act = nn.ReLU(inplace=True)

        ## pointwise convolution parameters
        self.pw_groups = 1
        self.pw_kernel_size = _pair(pw_kernel_size)
        self.pw_padding = pw_padding
        self.pw_stride = pw_stride
        self.pw_weight_shape = (self.out_channels, self.in_channels // self.pw_groups) + self.pw_kernel_size
        pw_weight_num_param = 1
        for wd in self.pw_weight_shape:
            pw_weight_num_param *= wd
        self.V_pw = nn.Parameter(torch.Tensor(e_exemplars, pw_weight_num_param))
        self.pw_bn = norm_layer(self.out_channels, **norm_kwargs)
        self.pw_act = nn.ReLU(inplace=False)

        ## Squeeze-and-excitation
        if se_ratio is not None and se_ratio > 0.:
            se_kwargs = resolve_se_args(se_kwargs, self.in_channels, nn.ReLU) #_get_activation_fn(se_act_layer))
            self.se = SqueezeExcite(self.in_channels, se_ratio=se_ratio, **se_kwargs)
        
        ## Implementation of Feedforward model after the QKV part
        self.linear1 = nn.Linear(self.out_channels, dim_feedforward)
        self.ff_dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, self.out_channels)
        self.norm1 = nn.LayerNorm(self.out_channels, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(self.out_channels, eps=layer_norm_eps)
        self.ff_dropout1 = nn.Dropout(ff_dropout)
        self.ff_dropout2 = nn.Dropout(ff_dropout)
        self.ff_activation = _get_activation_fn(ff_activation)

        # initialize the kernels 
        self.reset_parameters()

    def reset_parameters(self):
        init_weight_dw = get_initializer(
            partial(nn.init.kaiming_uniform_, a=math.sqrt(5)), self.e_exemplars, self.dw_weight_shape)
        init_weight_dw(self.V_dw)

        init_weight_pw = get_initializer(
            partial(nn.init.kaiming_uniform_, a=math.sqrt(5)), self.e_exemplars, self.pw_weight_shape)
        init_weight_pw(self.V_pw)
    
    def forward(self, x):

        residual = x
        
        # X: [B,C,H,W]

        # apply average pooler
        q = self.average_pooler(x)
        # d_k = q.shape[-1]
        # Q: [B,S,C]
        # ICRAFT NOTE:
        # 计算Keys外积的时候，K参数的转置需要提前算好，运行时不支持。这一步在顶层ET_Tracker.template函数里完成
        # outer product with keys
        #qk = einsum('b n c, k c -> b n k', q, self.K) # K^T: [C, K] QK^T: [B,S,K]
        # qk = torch.matmul(q, self.K.T)
        qk = torch.matmul(q, self.K_T)
        
        # if self.sm_norm:
        # qk = 1/math.sqrt(d_k) * qk
        qk = 1/16.0 * qk

        # apply softmax 
        attn = self.softmax(qk/self.temperature) # -> [batch_size, e_exemplars]
        
        # multiply attention map with values 
        #dw_qkv_kernel = einsum('b s k, k e -> b s e', attn, self.V_dw) # V: [K, E_dw]
        #pw_qkv_kernel = einsum('b s k, k e -> b s e', attn, self.V_pw) # V: [K, E_pw]
        dw_qkv_kernel = torch.matmul(attn, self.V_dw) # V: [K, E_dw]
        pw_qkv_kernel = torch.matmul(attn, self.V_pw) # V: [K, E_pw]

        ###########################################################################################
        ####### convolve input with the output instead of adding it to it in a residual way #######
        ###########################################################################################

        ## dw conv
        B, C, H, W = x.shape
        # ICRAFT NOTE:
        # 为了消除floor_divide算子，预先计算好dw_weight_shape
        # dw conv
        # dw_weight_shape = (B * self.out_channels, self.in_channels // self.dw_groups) + self.dw_kernel_size
        dw_weight = dw_qkv_kernel.view(192,1,5,5) #(dw_weight_shape)
        # ICRAFT NOTE:
        # 消除无效reshape
        # reshape the input
        # x = x.reshape(1,192,18,18) #(1, B * C, H, W)
        
        # apply convolution
        x = F.conv2d(x, dw_weight, bias=None, stride=1, padding=2,groups=192)
            # x, dw_weight, bias=None, stride=self.dw_stride, padding=self.dw_padding, 
            # groups=self.dw_groups * B)
        
        x = x.permute([1, 0, 2, 3]).view(1,192,18,18) #(B, self.out_channels, x.shape[-2], x.shape[-1])
        x = self.dw_bn(x)
        x = self.dw_act(x)

        ## SE
        x = self.se(x)

        ## pw conv
        B, C, H, W = x.shape
        # ICRAFT NOTE:
        # 为了消除floor_divide算子，预先计算好pw_weight_shape
        # dw conv
        # pw_weight_shape = (B * self.out_channels, self.in_channels // self.pw_groups) + self.pw_kernel_size
        pw_weight = pw_qkv_kernel.view(192,192,1,1) #(pw_weight_shape)
        # ICRAFT NOTE:
        # 消除无效view算子
        # reshape the input
        # x = x.view(1,192,18,18) #(1, B * C, H, W)
        
        # apply convolution
        x = F.conv2d(x, pw_weight, bias=None, stride=1, padding=0, groups=1)
            # x, pw_weight, bias=None, stride=self.pw_stride, padding=self.pw_padding, 
            # groups=self.pw_groups * B)
        
        x = x.permute([1, 0, 2, 3]).view(1,192,18,18) #(B, self.out_channels, x.shape[-2], x.shape[-1])
        x = self.pw_bn(x)
        x = self.pw_act(x)


        # if self.dropout:
        #     x = x + self.do(residual)
        # else:
        x = x + residual

        
        # reshape output of convolution operation
        # out = x.view(B, self.out_channels, -1).permute(0,2,1)
        out = x.view(1, 192, -1).permute(0,2,1)
        
        # FF network 
        out = self.norm1(out)
        out2 = self.linear2(self.ff_dropout(self.ff_activation(self.linear1(out))))
        out = out + self.ff_dropout2(out2)
        out = self.norm2(out)
        out = out.permute(0,2,1).view(1,192,18,18) #(B,C,H,W)
        
        return out


def Point_Neck_Mobile_simple_DP_forward(self, kernel, search):  #, stride_idx=None):
    '''stride_idx: 0 or 1. 0 represents stride 8. 1 represents stride 16'''
    # oup = {}
    corr_feat = self.pw_corr[0]([kernel], [search]) # [1,64,18,18]<-[[1,96,8,8]],[[1,96,18,18]]
    #print("corr_feat shape: ", corr_feat.shape)
    #print(f'type of corr_feat: {type(corr_feat)}')
    # if self.adjust:
    corr_feat = self.adj_layer[0](corr_feat) # [1,128,18,18]<-[1,64,18,18]
    # ICRAFT NOTE:
    # 将字典传递数据展开，更改接口
    # oup['cls'], oup['reg'] = corr_feat, corr_feat
    # return oup
    return corr_feat, corr_feat
Point_Neck_Mobile_simple_DP.forward = Point_Neck_Mobile_simple_DP_forward

def ET_Tracker__init__(self, linear_reg=True, 
                    search_size=256, 
                    template_size=128, 
                    stride=16, 
                    adj_channel=128, 
                    e_exemplars=4,
                    path_name='back_04502514044521042540+cls_211000022+reg_100000111_ops_32',
                    arch='LightTrackM_Subnet',
                    sm_normalization=False,
                    temperature=1,
                    dropout=False):
    super(ET_Tracker, self).__init__()

    '''
    Args:
        - sm_normalization: whether to normalize the QK^T by sqrt(C) in the MultiheadTransConver
    '''

    self.backbone_path_name = path_name

    # Backbone network
    siam_net = lighttrack_model.__dict__[arch](path_name, stride=stride)

    # Backbone
    self.backbone_net = siam_net.features

    # Neck
    self.neck = MC_BN(inp_c=[96])  # BN with multiple types of input channels

    # Feature Fusor
    self.feature_fusor = Point_Neck_Mobile_simple_DP(num_kernel_list=[64], matrix=True,
                                                            adj_channel=adj_channel)  # stride=8, stride=16

    inchannels = 128
    outchannels_cls = 256
    outchannels_reg = 192

    padding_3 = (3 - 1) // 2
    padding_5 = (5 - 1) // 2

    # ICRAFT NOTE:
    # 为了消除floor_divide算子等目的，增加4个ExemplerTransformer类
    self.cls_branch_1 = SeparableConv2d_BNReLU(inchannels, outchannels_cls, kernel_size=5, stride=1, padding=padding_5)
    # self.cls_branch_2 = ExemplarTransformer(in_channels=outchannels_cls, out_channels=outchannels_cls, dw_padding=padding_5, e_exemplars=e_exemplars, dw_kernel_size=5, pw_kernel_size=1, sm_normalization=sm_normalization, temperature=temperature, dropout=dropout)
    # self.cls_branch_3 = ExemplarTransformer(in_channels=outchannels_cls, out_channels=outchannels_cls, dw_padding=padding_3, e_exemplars=e_exemplars, dw_kernel_size=3, pw_kernel_size=1, sm_normalization=sm_normalization, temperature=temperature, dropout=dropout)
    # self.cls_branch_4 = ExemplarTransformer(in_channels=outchannels_cls, out_channels=outchannels_cls, dw_padding=padding_3, e_exemplars=e_exemplars, dw_kernel_size=3, pw_kernel_size=1, sm_normalization=sm_normalization, temperature=temperature, dropout=dropout)
    # self.cls_branch_5 = ExemplarTransformer(in_channels=outchannels_cls, out_channels=outchannels_cls, dw_padding=padding_3, e_exemplars=e_exemplars, dw_kernel_size=3, pw_kernel_size=1, sm_normalization=sm_normalization, temperature=temperature, dropout=dropout)
    self.cls_branch_2 = ExemplarTransformer256_5()
    self.cls_branch_3 = ExemplarTransformer256_3()
    self.cls_branch_4 = ExemplarTransformer256_3()
    self.cls_branch_5 = ExemplarTransformer256_3()
    self.cls_branch_6 = SeparableConv2d_BNReLU(outchannels_cls, outchannels_cls, kernel_size=3, stride=1, padding=padding_3)
    self.cls_pred_head = cls_pred_head(inchannels=outchannels_cls)

    self.bbreg_branch_1 = SeparableConv2d_BNReLU(inchannels, outchannels_reg, kernel_size=3, stride=1, padding=padding_3)
    # self.bbreg_branch_2 = ExemplarTransformer(in_channels=outchannels_reg, out_channels=outchannels_reg, dw_padding=padding_3, e_exemplars=e_exemplars, dw_kernel_size=3, pw_kernel_size=1, sm_normalization=sm_normalization, temperature=temperature, dropout=dropout)
    # self.bbreg_branch_3 = ExemplarTransformer(in_channels=outchannels_reg, out_channels=outchannels_reg, dw_padding=padding_3, e_exemplars=e_exemplars, dw_kernel_size=3, pw_kernel_size=1, sm_normalization=sm_normalization, temperature=temperature, dropout=dropout)
    # self.bbreg_branch_4 = ExemplarTransformer(in_channels=outchannels_reg, out_channels=outchannels_reg, dw_padding=padding_3, e_exemplars=e_exemplars, dw_kernel_size=3, pw_kernel_size=1, sm_normalization=sm_normalization, temperature=temperature, dropout=dropout)
    # self.bbreg_branch_5 = ExemplarTransformer(in_channels=outchannels_reg, out_channels=outchannels_reg, dw_padding=padding_3, e_exemplars=e_exemplars, dw_kernel_size=3, pw_kernel_size=1, sm_normalization=sm_normalization, temperature=temperature, dropout=dropout)
    # self.bbreg_branch_6 = ExemplarTransformer(in_channels=outchannels_reg, out_channels=outchannels_reg, dw_padding=padding_5, e_exemplars=e_exemplars, dw_kernel_size=5, pw_kernel_size=1, sm_normalization=sm_normalization, temperature=temperature, dropout=dropout)
    # self.bbreg_branch_7 = ExemplarTransformer(in_channels=outchannels_reg, out_channels=outchannels_reg, dw_padding=padding_5, e_exemplars=e_exemplars, dw_kernel_size=5, pw_kernel_size=1, sm_normalization=sm_normalization, temperature=temperature, dropout=dropout)
    self.bbreg_branch_2 = ExemplarTransformer192_3()
    self.bbreg_branch_3 = ExemplarTransformer192_3()
    self.bbreg_branch_4 = ExemplarTransformer192_3()
    self.bbreg_branch_5 = ExemplarTransformer192_3()
    self.bbreg_branch_6 = ExemplarTransformer192_5()
    self.bbreg_branch_7 = ExemplarTransformer192_5()
    self.bbreg_branch_8 = SeparableConv2d_BNReLU(outchannels_reg, outchannels_reg, kernel_size=5, stride=1, padding=padding_5)
    self.reg_pred_head = reg_pred_head(inchannels=outchannels_reg, linear_reg=linear_reg)

def ET_Tracker_template(self, z):
    '''
    Used during the tracking -> computes the embedding of the target in the first frame.
    '''
    # ICRAFT NOTE:
    # 提前做好K参数的转置
    self.cls_branch_2.K_T = self.cls_branch_2.K.T.contiguous().detach()
    self.cls_branch_3.K_T = self.cls_branch_3.K.T.contiguous().detach()
    self.cls_branch_4.K_T = self.cls_branch_4.K.T.contiguous().detach()
    self.cls_branch_5.K_T = self.cls_branch_5.K.T.contiguous().detach()
    self.bbreg_branch_2.K_T = self.bbreg_branch_2.K.T.contiguous().detach()
    self.bbreg_branch_3.K_T = self.bbreg_branch_3.K.T.contiguous().detach()
    self.bbreg_branch_4.K_T = self.bbreg_branch_4.K.T.contiguous().detach()
    self.bbreg_branch_5.K_T = self.bbreg_branch_5.K.T.contiguous().detach()
    self.bbreg_branch_6.K_T = self.bbreg_branch_6.K.T.contiguous().detach()
    self.bbreg_branch_7.K_T = self.bbreg_branch_7.K.T.contiguous().detach()
    with torch.no_grad():
        # ICRAFT NOTE:
        # 将template网络导出为pt
        t_z = torch.randn((1,3,127,127))
        # t_z.numpy().astype(np.float32).tofile('./ettrack/template_1_3_127_127.ftmp')#GPU环境下导出为handtanh
        ettrack_template_backbone_t = torch.jit.trace(self.backbone_net, t_z)# cpu环境下会将handtanh变为relu6
        torch.jit.save(ettrack_template_backbone_t,TRACE_PATH + 'ettrack_net1_1x3x127x127_traced.pt')
        print('net1 traced')
        # ICRAFT NOTE:
        # 将z和zf导出ftmp，用于构建量化校准集
        # z.cpu().contiguous().numpy().astype(np.float32).tofile('icraft/calibration/airplane-1_z.ftmp')
        # self.zf.cpu().contiguous().numpy().astype(np.float32).tofile('icraft/calibration/airplane-1_zf.ftmp')
        self.zf = self.backbone_net(z) # [1,96, 8, 8]

# ICRAFT NOTE:
# 因为ET_Tracker.forward函数中zf是模板计算好后的特征，为了导出CNN+TFM网络需要增加zf输入
def ET_Tracker_forward(self, x, zf):
    # [1,3,288,288]
    xf = self.backbone_net(x)
    # [1,96,16,16]
    # Batch Normalization before Corr
    # ICRAFT NOTE:
    # 为了部署，将成员变量作为前向输入
    # zf, xf = self.neck(self.zf, xf)  #[1,96,8,8] [1,96,16,16]<-[1,96,8,8] [1,96,16,16]
    zf, xf = self.neck(zf, xf)  #[1,96,8,8] [1,96,16,16]<-[1,96,8,8] [1,96,16,16]

    # feat_dict = self.feature_fusor(zf, xf) # cls:[1,128,16,16],[1,128,16,16]<-[1,96,8,8] [1,96,16,16]
    feat_cls, feat_reg = self.feature_fusor(zf, xf) # cls:[1,128,16,16],[1,128,16,16]<-[1,96,8,8] [1,96,16,16]

    c = self.cls_branch_1(feat_cls)#(feat_dict['cls'])
    c = self.cls_branch_2(c) 
    c = self.cls_branch_3(c) 
    c = self.cls_branch_4(c) 
    c = self.cls_branch_5(c) 
    c = self.cls_branch_6(c)
    c = self.cls_pred_head(c) # [1,1,16,16]
    
    b = self.bbreg_branch_1(feat_reg)#(feat_dict['reg'])
    b = self.bbreg_branch_2(b) 
    b = self.bbreg_branch_3(b) 
    b = self.bbreg_branch_4(b) 
    b = self.bbreg_branch_5(b) 
    b = self.bbreg_branch_6(b) 
    b = self.bbreg_branch_7(b) 
    b = self.bbreg_branch_8(b)
    b = self.reg_pred_head(b) # [1,4,16,16]
    return c, b

ET_Tracker.__init__ = ET_Tracker__init__
ET_Tracker.template = ET_Tracker_template
ET_Tracker.forward = ET_Tracker_forward

from lib.utils.utils import get_subwindow_tracking, python2round
from lib.utils.utils import cxy_wh_2_rect, get_axis_aligned_bbox
from pytracking.tracker.et_tracker.et_tracker import Config
# ICRAFT NOTE:
# 重新定义initialize改变流程
def TransconverTracker_initialize(self, image, info: dict) -> dict:
    ''' initialize the model '''

    state_dict = dict()

    # Initialize some stuff
    self.frame_num = 1
    if not self.params.has('device'):
        self.params.device = 'cuda' if self.params.use_gpu else 'cpu'

    self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Initialize network
    # verify that the model is correctly initialized:
    self.initialize_features()

    # The Baseline network
    self.net = self.params.net
    self.net.eval()
    self.net.to(self.params.device)

    self.weight_style = self.params.get('weight_style', 'regular')
    print(f'tracker weight style: {self.weight_style}')

    # Time initialization
    tic = time.time()

    # Get target position and size
    state = torch.tensor(info['init_bbox']) # x,y,w,h
    cx, cy, w, h = get_axis_aligned_bbox(state)
    self.target_pos = np.array([cx,cy])
    self.target_sz = np.array([w,h])
    #self.target_pos = torch.Tensor([state[1] + (state[3] - 1)/2, state[0] + (state[2] - 1)/2])
    #self.target_sz = torch.Tensor([state[3], state[2]])

    # Get object id
    self.object_id = info.get('object_ids', [None])[0]
    self.id_str = '' if self.object_id is None else ' {}'.format(self.object_id)

    # Set sizes
    self.image_sz = torch.Tensor([image.shape[0], image.shape[1]])
    sz = self.params.image_sample_size # search size (256, 256)
    sz = torch.Tensor([sz, sz] if isinstance(sz, int) else sz)
    self.img_sample_sz = sz
    self.img_support_sz = self.img_sample_sz
    self.stride = self.params.stride

    # LightTrack specific parameters
    p = Config(stride=self.stride, even=self.params.even)

    state_dict['im_h'] = image.shape[0]
    state_dict['im_w'] = image.shape[1]
    # ICRAFT NOTE:
    # 原来流程会根据初始框大小在原始帧占比决定网格是按照18x18（小）还是16x16（大），统一改为18x18
    # if ((self.target_sz[0] * self.target_sz[1]) / float(state_dict['im_h'] * state_dict['im_w'])) < 0.004:
    #     p.instance_size = self.params.big_sz # cfg_benchmark['big_sz']  # -> p.instance_size = 288
    #     p.renew()
    # else:
    #     p.instance_size = self.params.small_sz # cfg_benchmark['small_sz'] # -> p.instance_size = 256
    #     p.renew()
    ### ICRAFT NOTE: Force to use big_sz 288
    p.instance_size = self.params.big_sz # cfg_benchmark['big_sz']  # -> p.instance_size = 288
    p.renew()

    # compute grids
    self.grids(p)

    wc_z = self.target_sz[0] + p.context_amount * sum(self.target_sz)
    hc_z = self.target_sz[1] + p.context_amount * sum(self.target_sz)
    s_z = round(np.sqrt(wc_z * hc_z).item())

    avg_chans = np.mean(image, axis=(0, 1))
    z_crop, _ = get_subwindow_tracking(image, self.target_pos, p.exemplar_size, s_z, avg_chans)
    z_crop = self.normalize(z_crop)
    z = z_crop.unsqueeze(0)
    self.net.template(z.to(self.params.device))

    if p.windowing == 'cosine':
        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))  # [17,17]
    elif p.windowing == 'uniform':
        window = np.ones(int(p.score_size), int(p.score_size))
    else:
        raise ValueError("Unsupported window type")

    state_dict['p'] = p
    state_dict['avg_chans'] = avg_chans
    state_dict['window'] = window
    state_dict['target_pos'] = self.target_pos
    state_dict['target_sz'] = self.target_sz
    state_dict['time'] = time.time() - tic
    return state_dict

def TransconverTracker_update(self, x_crops, target_pos, target_sz, window, scale_z, p, debug=False, writer=None):
    with torch.no_grad():
        # ICRAFT NOTE:
        # 导出到pt和onnx的代码
        # ICRAFT NOTE: PT
        x = torch.randn((1,3,288,288))#.cuda()
        zf = torch.randn((1,96,8,8))#.cuda()
        # x.numpy().astype(np.float32).tofile('icraft/search_1_3_288_288.ftmp')
        # zf.numpy().astype(np.float32).tofile('icraft/template_1_96_8_8.ftmp')
        global TRACE
        if TRACE:
            traced = torch.jit.trace(self.net, [x, zf])
            torch.jit.save(traced, TRACE_PATH +'ettrack_net2_1x3x288x288_traced.pt')
            print('net2 traced')
            TRACE = False
            sys.exit()
        cls_score, bbox_pred = self.net.forward(x_crops.to(self.params.device), self.net.zf)

    # ICRAFT NOTE:
    # 导出输入作为量化校准集
    # if self.frame_num in [2,502, 1002, 1502, 2002, 2502]:
    #     x_crops.cpu().contiguous().numpy().astype(np.float32).tofile(f'icraft/calibration/x_{self.frame_num}.ftmp')
    # to numpy on cpu
    cls_score = torch.sigmoid(cls_score).squeeze().cpu().data.numpy() #[18,18]<-[1,1,18,18]

    # bbox to real predict
    bbox_pred = bbox_pred.squeeze().cpu().data.numpy()#[4,18,18]<-[1,4,18,18]

    pred_x1 = self.grid_to_search_x - bbox_pred[0, ...]
    pred_y1 = self.grid_to_search_y - bbox_pred[1, ...]
    pred_x2 = self.grid_to_search_x + bbox_pred[2, ...]
    pred_y2 = self.grid_to_search_y + bbox_pred[3, ...]       
    

    # size penalty
    s_c = self.change(self.sz(pred_x2 - pred_x1, pred_y2 - pred_y1) / (self.sz_wh(target_sz)))  # scale penalty
    r_c = self.change((target_sz[0] / target_sz[1]) / ((pred_x2 - pred_x1) / (pred_y2 - pred_y1)))  # ratio penalty

    penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)
    pscore = penalty * cls_score
    
    # window penalty
    pscore = pscore * (1 - p.window_influence) + window * p.window_influence
    
    # get max
    r_max, c_max = np.unravel_index(pscore.argmax(), pscore.shape)
    
    # to real size
    pred_x1 = pred_x1[r_max, c_max]
    pred_y1 = pred_y1[r_max, c_max]
    pred_x2 = pred_x2[r_max, c_max]
    pred_y2 = pred_y2[r_max, c_max]

    pred_xs = (pred_x1 + pred_x2) / 2
    pred_ys = (pred_y1 + pred_y2) / 2
    pred_w = pred_x2 - pred_x1
    pred_h = pred_y2 - pred_y1

    diff_xs = pred_xs - p.instance_size // 2
    diff_ys = pred_ys - p.instance_size // 2
    
    diff_xs, diff_ys, pred_w, pred_h = diff_xs / scale_z, diff_ys / scale_z, pred_w / scale_z, pred_h / scale_z

    target_sz = target_sz / scale_z

    # size learning rate
    lr = penalty[r_max, c_max] * cls_score[r_max, c_max] * p.lr
    print(lr)
    # size rate
    res_xs = target_pos[0] + diff_xs
    res_ys = target_pos[1] + diff_ys
    res_w = pred_w * lr + (1 - lr) * target_sz[0]
    res_h = pred_h * lr + (1 - lr) * target_sz[1]

    target_pos = np.array([res_xs, res_ys])
    target_sz = target_sz * (1 - lr) + lr * np.array([res_w, res_h])

    if debug:
        return target_pos, target_sz, cls_score[r_max, c_max], cls_score
    else:
        return target_pos, target_sz, cls_score[r_max, c_max]

TransconverTracker.initialize = TransconverTracker_initialize
TransconverTracker.update = TransconverTracker_update

# ICRAFT NOTE:
# 改变Tracker的get_parameters和create_tracker方法，这样可以调用修改后的模块构造网络
def Tracker_get_parameters(self):
    params = TrackerParams()

    params.debug = 0
    params.visualization = False

    params.use_gpu = True

    params.checkpoint_epoch = 35

    params.net = ET_Tracker(search_size=256,
                            template_size=128,
                            stride=16,
                            e_exemplars=4,
                            sm_normalization=True, 
                            temperature=2,
                            dropout=False)

    params.big_sz = 288
    params.small_sz = 256
    params.stride = 16
    params.even = 0
    params.model_name = 'et_tracker'

    params.image_sample_size = 256
    params.image_template_size = 128
    params.search_area_scale = 5

    params.window_influence = 0
    params.lr = 0.616
    params.penalty_k = 0.007
    params.context_amount = 0.5

    params.features_initialized = False

    return params

def Tracker_create_tracker(self, params):
    t = TransconverTracker(params)
    t.visdom = self.visdom
    return t

Tracker.get_parameters = Tracker_get_parameters
Tracker.create_tracker = Tracker_create_tracker

TRACE_PATH = "../2_compile/fmodel/"
os.makedirs(os.path.dirname(TRACE_PATH), exist_ok=True)
TRACE = True

if __name__ == '__main__':
    dataset_name = 'lasot'
    tracker_name = 'et_tracker'
    tracker_param = 'et_tracker'
    visualization=None
    debug=None
    visdom_info=None
    run_id = 2405101501
    dataset = get_dataset(dataset_name)

    tracker = Tracker(tracker_name, tracker_param, run_id)
    # et_tracker构造函数在此函数内调用
    params = tracker.get_parameters()
    visualization_ = visualization

    debug_ = debug
    if debug is None:
        debug_ = getattr(params, 'debug', 0)
    if visualization is None:
        if debug is None:
            visualization_ = getattr(params, 'visualization', False)
        else:
            visualization_ = True if debug else False

    params.visualization = visualization_
    params.debug = debug_
    params.use_gpu = False
    
    for seq in dataset[:]:
        print(seq)
        def _results_exist():
            if seq.dataset == 'oxuva':
                vid_id, obj_id = seq.name.split('_')[:2]
                pred_file = os.path.join(tracker.results_dir, '{}_{}.csv'.format(vid_id, obj_id))
                return os.path.isfile(pred_file)
            elif seq.object_ids is None:
                bbox_file = '{}/{}.txt'.format(tracker.results_dir, seq.name)
                return os.path.isfile(bbox_file)
            else:
                bbox_files = ['{}/{}_{}.txt'.format(tracker.results_dir, seq.name, obj_id) for obj_id in seq.object_ids]
                missing = [not os.path.isfile(f) for f in bbox_files]
                return sum(missing) == 0

        visdom_info = {} if visdom_info is None else visdom_info

        if _results_exist() and not debug:
            print('FPS: {}'.format(-1))
            continue

        print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq.name))

        tracker._init_visdom(visdom_info, debug_)
        if visualization_ and tracker.visdom is None:
            tracker.init_visualization()

        # Get init information
        init_info = seq.init_info()
        et_tracker = tracker.create_tracker(params)
        output = {'target_bbox': [],
            'time': [],
            'segmentation': [],
            'object_presence_score': []}

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = tracker._read_image(seq.frames[0])

        if et_tracker.params.visualization and tracker.visdom is None:
            tracker.visualize(image, init_info.get('init_bbox'))

        start_time = time.time()
        out = et_tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)

        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time,
                        'segmentation': init_info.get('init_mask'),
                        'object_presence_score': 1.}

        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = tracker._read_image(frame_path)

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            out = et_tracker.track(image, info)
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

            segmentation = out['segmentation'] if 'segmentation' in out else None
            if tracker.visdom is not None:
                tracker.visdom_draw_tracking(image, out['target_bbox'], segmentation)
            elif et_tracker.params.visualization:
                tracker.visualize(image, out['target_bbox'], segmentation)

        for key in ['target_bbox', 'segmentation']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        output['image_shape'] = image.shape[:2]
        output['object_presence_score_threshold'] = et_tracker.params.get('object_presence_score_threshold', 0.55)

        sys.stdout.flush()

        if isinstance(output['time'][0], (dict, OrderedDict)):
            exec_time = sum([sum(times.values()) for times in output['time']])
            num_frames = len(output['time'])
        else:
            exec_time = sum(output['time'])
            num_frames = len(output['time'])

        print('FPS: {}'.format(num_frames / exec_time))

        if not debug:
            _save_tracker_output(seq, tracker, output)