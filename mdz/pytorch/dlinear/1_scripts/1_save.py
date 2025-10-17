# -*- coding:utf-8 -*-
# 该脚本用来保存torchscript模型
import argparse
import time
from pathlib import Path

import cv2
import torch
from numpy import random
import sys 
sys.path.append(R"../0_dlinear")

import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
from utils.tools import visual

from models import DLinear
import torch.nn.functional as F

import random
import numpy as np
def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def cal_precision_recall(predictions, trues):
    TP = np.sum((predictions == 1) & (trues == 1))
    FP = np.sum((predictions == 1) & (trues == 0))
    FN = np.sum((predictions == 0) & (trues == 1))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1
def segrnn_forward_ltf(self, x,pos_emb,hx):

    x = self.valueEmbedding(x.reshape(-1, self.seg_num_x, self.seg_len))
    # encoding
    _, hn = self.rnn(x, hx) # bc,n,d  1,bc,d
    h0 = F.max_pool1d(hn, kernel_size=1, stride=1)
    h1 = F.max_pool1d(hn, kernel_size=1, stride=1)
    h2 = F.max_pool1d(hn, kernel_size=1, stride=1)
    h3 = F.max_pool1d(hn, kernel_size=1, stride=1)
    hn = torch.cat([h0,h1,h2,h3], dim=2)
    hn = hn.view(1, -1, self.d_model)
    _, hy = self.rnn(pos_emb, hn) # bcm,1,d  1,bcm,d
    y = self.predict(hy).view(-1, self.enc_in, self.pred_len)
    y = y.permute(0, 2, 1)
    return y

def dlinear_forward_ltf(self, x_enc):
    # Encoder
    return self.encoder(x_enc)
def dlinear_forward_cls(self, x_enc):
    # Encoder
    enc_out = self.encoder(x_enc)
    # Output
    # (batch_size, seq_length * d_model)
    output = enc_out.reshape(enc_out.shape[0], -1)
    # (batch_size, num_classes)
    output = self.projection(output)
    return output


def infer(self, setting, test=0):
    
    if test:
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join('../0_dlinear/checkpoints/' + setting, 'checkpoint.pth')))

    preds = []
    trues = []
    folder_path = './test_results/' + setting + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    self.model.eval()
    with torch.no_grad():
        if self.args.task_name == 'long_term_forecast':
            test_data, test_loader = self._get_data(flag='test')
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                #forward
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                #trace
                DLinear.Model.forward = dlinear_forward_ltf


                traced_script_module = torch.jit.trace(self.model.cpu(),batch_x.cpu())
                traced_out = traced_script_module(batch_x)


                # 保存TorchScript模型
                traced_script_module.save(R'../2_compile/fmodel/dlinear_ltf.pt')
                torch.onnx.export(self.model, (batch_x.cpu()), R"../2_compile/fmodel/dlinear_ltf.onnx", verbose=True)

                break 
        elif args.task_name == 'classification':
            test_data, test_loader = self._get_data(flag='TEST')    
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                #forward
                # outputs = self.model(batch_x, padding_mask, None, None)
                #trace

                DLinear.Model.forward = dlinear_forward_cls

                # outputs = self.model(batch_x, padding_mask, hx)


                traced_script_module = torch.jit.trace(self.model.cpu(),(batch_x.cpu()))
                traced_out = traced_script_module(batch_x.cpu())
                outputs = traced_out

                # 保存TorchScript模型
                traced_script_module.save(R'../2_compile/fmodel/dlinear_cls.pt')
                torch.onnx.export(self.model, (batch_x.cpu()), R"../2_compile/fmodel/dlinear_cls.onnx", verbose=True)

                # probs = torch.nn.functional.softmax(outputs)  # (total_samples, num_classes) est. prob. for each class and sample
                # predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
                # trues = label.flatten().cpu().numpy()

                # print('predictions:{},trues{}'.format(predictions,trues))

                break
            #     preds.append(outputs.detach())
            #     trues.append(label)

            # preds = torch.cat(preds, 0)
            # trues = torch.cat(trues, 0)
            # print('test shape:', preds.shape, trues.shape)

            # probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
            # predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
            # trues = trues.flatten().cpu().numpy()
            # accuracy = cal_accuracy(predictions, trues)
            # p,r,f1 = cal_precision_recall(predictions, trues)

            # # result save
            # folder_path = './results/' + setting + '/'
            # if not os.path.exists(folder_path):
            #     os.makedirs(folder_path)

            # print('accuracy:{}'.format(accuracy))
            # print('p:{}'.format(p))
            # print('r:{}'.format(r))
            # print('f1:{}'.format(f1))




    return
if __name__ == '__main__':
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)



    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=48,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False, 
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')
    
    args = parser.parse_args()

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'long_term_forecast':
        Exp_Long_Term_Forecast.infer = infer
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'classification':
        Exp_Classification.infer = infer
        Exp = Exp_Classification
    else: 
        Exp = Exp_Long_Term_Forecast
        

    if args.is_training:
        pass
    else:
        ii = 0
        exp = Exp(args)  # set experiments
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)
        print('>>>>>>>infer : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.infer(setting, test=1)

