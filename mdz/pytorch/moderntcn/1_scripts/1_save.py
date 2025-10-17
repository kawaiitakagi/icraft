import argparse
import os
import sys
import torch
import torch.nn.functional as F


import random
import numpy as np

# from utils.str2bool import str2bool
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
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

def pre_process(x,te = None):
    # x = x.permute(0, 2, 1)
    if te is not None:
        te = te.permute(0, 2, 1)
    # x = x.permute(0, 2, 1)
    #_get_statistics
    dim2reduce = tuple(range(1, x.ndim-1))
    eps=1e-5
    mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
    stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + eps).detach()
    #normalize
    x = x - mean
    x = x / stdev

    x = x.permute(0, 2, 1)

    return x,te,mean,stdev
def moderntcn_forwrad(self, x):
    if self.task_name == 'classification':
        x = self.classification(x)
    return x
def moderntcn_forwrad_ltf(self,x,mean,stdev):
    # x = self.forward_feature(x,te)
    # x = self.head(x)
    x = self.forward_feature(x)
    x = self.head(x)
    x = x.permute(0, 2, 1)
    x = x * stdev
    x = x + mean
    return x
def moderntcn_forwrad_cls(self, x):
    x = x.permute(0, 2, 1)
    te = None
    x = self.model(x)
    return x


def forward_feature_ltf(self, x, te=None):

    B,M,L=x.shape

    x = x.unsqueeze(-2)
    for i in range(self.num_stage):
        B, M, D, N = x.shape
        x = x.reshape(B * M, D, N)
        if i==0:
            if self.patch_size != self.patch_stride:
                # stem layer padding
                pad_len = self.patch_size - self.patch_stride

                hn = x[:,:,335:]
                h0 = F.max_pool1d(hn, kernel_size=1, stride=1)
                h1 = F.max_pool1d(hn, kernel_size=1, stride=1)
                h2 = F.max_pool1d(hn, kernel_size=1, stride=1)
                h3 = F.max_pool1d(hn, kernel_size=1, stride=1)

                pad = torch.cat([h0,h1,h2,h3], dim=2)
                # pad = x[:,:,-1:].repeat(1,1,pad_len)
                x = torch.cat([x,pad],dim=-1)
        else:
            if N % self.downsample_ratio != 0:
                pad_len = self.downsample_ratio - (N % self.downsample_ratio)
                x = torch.cat([x, x[:, :, -pad_len:]],dim=-1)
        x = self.downsample_layers[i](x)
        _, D_, N_ = x.shape
        x = x.reshape(B, M, D_, N_)
        x = self.stages[i](x)
    return x

def forward_feature_cls(self, x, te=None):

    B,M,L=x.shape

    x = x.unsqueeze(-2)

    for i in range(self.num_stage): 
        B, M, D, N = x.shape
        x = x.reshape(B * M, D, N)
        if i==0:
            if self.patch_size != self.patch_stride:
                # stem layer padding
                pad_len = self.patch_size - self.patch_stride
                hn = x[:,:,404:]
                h0 = F.max_pool1d(hn, kernel_size=1, stride=1)
                h1 = F.max_pool1d(hn, kernel_size=1, stride=1)
                h2 = F.max_pool1d(hn, kernel_size=1, stride=1)
                h3 = F.max_pool1d(hn, kernel_size=1, stride=1)

                pad = torch.cat([h0,h1,h2,h3], dim=2)

                #org_pad
                # pad = x[:,:,-1:].repeat(1,1,pad_len)
                x = torch.cat([x,pad],dim=-1)
        else:
            if N % self.downsample_ratio != 0:
                pad_len = self.downsample_ratio - (N % self.downsample_ratio)
                x = torch.cat([x, x[:, :, -pad_len:]],dim=-1)
        x = self.downsample_layers[i](x)
        _, D_, N_ = x.shape
        x = x.reshape(B, M, D_, N_)
        x = self.stages[i](x)
    return x



def infer_ltf(self, setting, test=0):
    test_data, test_loader = self._get_data(flag='test')

    if test:
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join('../weights/' + setting, 'checkpoint.pth')))

    preds = []
    trues = []
    inputx = []
    folder_path = './test_results/' + setting + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    self.model.eval()
    if self.args.call_structural_reparam and hasattr(self.model, 'structural_reparam'):
        self.model.structural_reparam()

    traced_script_module = None
    os.makedirs('../2_compile/qtset/ltf/',exist_ok=True)
    os.makedirs('../3_deploy/modelzoo/moderntcn/io/ltf/input/',exist_ok=True)
    os.makedirs('../3_deploy/modelzoo/moderntcn/io/ltf/label/',exist_ok=True)
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)




            # encoder - decoder
            if self.args.use_amp:
                pass
            else:
                if 'Linear' in self.args.model or 'TST' in self.args.model:
                    outputs = self.model(batch_x)
                elif 'TCN' in self.args.model:

                    if traced_script_module is None:
                        x,te = batch_x.clone(), batch_x_mark.clone()

                        ModernTCN.ModernTCN.forward = moderntcn_forwrad_ltf
                        ModernTCN.ModernTCN.forward_feature = forward_feature_ltf

                        x,te,mean,stdev = pre_process(x,te)


                        traced_script_module = torch.jit.trace(self.model.model.cpu(),(x.cpu(),mean.cpu(),stdev.cpu()))

                        #  保存TorchScript模型
                        traced_script_module.save('../2_compile/fmodel/moderntcn_ltf.pt')
                        torch.onnx.export(self.model.model.cpu(), (x.cpu(),mean.cpu(),stdev.cpu()), '../2_compile/fmodel/moderntcn_ltf.onnx', verbose=True)

                        outputs = traced_script_module(x.cpu(),mean.cpu(),stdev.cpu())
                        # x = x.permute(0, 2, 1)
                        # x = x * stdev
                        # x = x + mean
                        # outputs = x
                        x.cpu().numpy().astype('float32').tofile('../2_compile/qtset/ltf/'+str(i)+'.ftmp')
                        mean.cpu().numpy().astype('float32').tofile('../2_compile/qtset/ltf/'+str(i)+'_mean.ftmp')
                        stdev.cpu().numpy().astype('float32').tofile('../2_compile/qtset/ltf/'+str(i)+'_stdev.ftmp')

                        x.cpu().numpy().astype('float32').tofile('../3_deploy/modelzoo/moderntcn/io/ltf/input/'+str(i)+'.ftmp')
                        mean.cpu().numpy().astype('float32').tofile('../3_deploy/modelzoo/moderntcn/io/ltf/input/'+str(i)+'_mean.ftmp')
                        stdev.cpu().numpy().astype('float32').tofile('../3_deploy/modelzoo/moderntcn/io/ltf/input/'+str(i)+'_stdev.ftmp')
                        batch_y.cpu().numpy().astype('float32').tofile('../3_deploy/modelzoo/moderntcn/io/ltf/label/'+str(i)+'.ftmp') 
                    else:
                        x,te = batch_x.clone(), batch_x_mark.clone()

                        x,te,mean,stdev = pre_process(x,te)
                        x.cpu().numpy().astype('float32').tofile('../2_compile/qtset/ltf/'+str(i)+'.ftmp')
                        mean.cpu().numpy().astype('float32').tofile('../2_compile/qtset/ltf/'+str(i)+'_mean.ftmp')
                        stdev.cpu().numpy().astype('float32').tofile('../2_compile/qtset/ltf/'+str(i)+'_stdev.ftmp')

                        x.cpu().numpy().astype('float32').tofile('../3_deploy/modelzoo/moderntcn/io/ltf/input/'+str(i)+'.ftmp')
                        mean.cpu().numpy().astype('float32').tofile('../3_deploy/modelzoo/moderntcn/io/ltf/input/'+str(i)+'_mean.ftmp')
                        stdev.cpu().numpy().astype('float32').tofile('../3_deploy/modelzoo/moderntcn/io/ltf/input/'+str(i)+'_stdev.ftmp')
                        batch_y.cpu().numpy().astype('float32').tofile('../3_deploy/modelzoo/moderntcn/io/ltf/label/'+str(i)+'.ftmp') 
            # break

    return

def infer_cls(self, setting, test=0):
    test_data, test_loader = self._get_data(flag='TEST')
    if test:
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join('../weights/' + setting, 'checkpoint.pth')))

    preds = []
    trues = []
    folder_path = './test_results/' + setting + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    self.model.eval()
    traced_script_module = None

    os.makedirs('../2_compile/qtset/cls/',exist_ok=True)
    os.makedirs('../3_deploy/modelzoo/moderntcn/io/cls/input/',exist_ok=True)
    os.makedirs('../3_deploy/modelzoo/moderntcn/io/cls/label/',exist_ok=True)
    with torch.no_grad():
        for i, (batch_x, label, padding_mask) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            padding_mask = padding_mask.float().to(self.device)
            label = label.to(self.device)

            # outputs = self.model(batch_x, padding_mask, None, None)


            # Model.forward = m_forward
            if traced_script_module is None:

                ModernTCN.ModernTCN.forward = moderntcn_forwrad
                ModernTCN.Model.forward = moderntcn_forwrad_cls
                ModernTCN.ModernTCN.forward_feature = forward_feature_cls

                traced_script_module = torch.jit.trace(self.model.cpu(),batch_x.cpu())

                outputs = traced_script_module(batch_x.cpu())
                traced_script_module.save('../2_compile/fmodel/moderntcn_cls.pt')
                
                torch.onnx.export(self.model.cpu(), batch_x.cpu(), '../2_compile/fmodel/moderntcn_cls.onnx', verbose=True)

                batch_x.cpu().numpy().astype('float32').tofile('../2_compile/qtset/cls/'+str(i)+'.ftmp')
                batch_x.cpu().numpy().astype('float32').tofile('../3_deploy/modelzoo/moderntcn/io/cls/input/'+str(i)+'.ftmp')
                label.cpu().numpy().astype('float32').tofile('../3_deploy/modelzoo/moderntcn/io/cls/label/'+str(i)+'.ftmp')  


            else:
                outputs = traced_script_module(batch_x.cpu())
                batch_x.cpu().numpy().astype('float32').tofile('../2_compile/qtset/cls/'+str(i)+'.ftmp')
                batch_x.cpu().numpy().astype('float32').tofile('../3_deploy/modelzoo/moderntcn/io/cls/input/'+str(i)+'.ftmp')
                label.cpu().numpy().astype('float32').tofile('../3_deploy/modelzoo/moderntcn/io/cls/label/'+str(i)+'.ftmp')  

        return




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ModernTCN')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='ModernTCN',
                        help='model name, options: [ModernTCN]')

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
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')


    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')



    #ModernTCN
    parser.add_argument('--stem_ratio', type=int, default=6, help='stem ratio')
    parser.add_argument('--downsample_ratio', type=int, default=2, help='downsample_ratio')
    parser.add_argument('--ffn_ratio', type=int, default=2, help='ffn_ratio')
    parser.add_argument('--patch_size', type=int, default=16, help='the patch size')
    parser.add_argument('--patch_stride', type=int, default=8, help='the patch stride')

    parser.add_argument('--num_blocks', nargs='+',type=int, default=[1,1,1,1], help='num_blocks in each stage')
    parser.add_argument('--large_size', nargs='+',type=int, default=[31,29,27,13], help='big kernel size')
    parser.add_argument('--small_size', nargs='+',type=int, default=[5,5,5,5], help='small kernel size for structral reparam')
    parser.add_argument('--dims', nargs='+',type=int, default=[256,256,256,256], help='dmodels in each stage')
    parser.add_argument('--dw_dims', nargs='+',type=int, default=[256,256,256,256], help='dw dims in dw conv in each stage')

    parser.add_argument('--small_kernel_merged', type=str2bool, default=False, help='small_kernel has already merged or not')
    parser.add_argument('--call_structural_reparam', type=bool, default=False, help='structural_reparam after training')
    parser.add_argument('--use_multi_scale', type=str2bool, default=True, help='use_multi_scale fusion')


    # PatchTST
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # Formers
    parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
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
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')

    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    #multi task
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                            help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # classfication task
    parser.add_argument('--class_dropout', type=float, default=0.05, help='classfication dropout')

    args = parser.parse_args()

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    # Exp = Exp_Main
    if args.task_name == 'long_term_forecast':
        sys.path.append(R"../0_moderntcn/ModernTCN-Long-term-forecasting/")
        from exp.exp_ModernTCN import Exp_Main
        Exp_Main.infer = infer_ltf
        Exp = Exp_Main
        from data_provider.data_factory import data_provider

        from exp.exp_basic import Exp_Basic
        from models import ModernTCN
        # from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
        # from utils.metrics import metric

        import numpy as np
        import torch

        import os

        import matplotlib.pyplot as plt
        import numpy as np
    elif args.task_name == 'classification':
        sys.path.append(R"../0_moderntcn/ModernTCN-classification/")
        from exp.exp_classification import Exp_Classification
        from exp.exp_basic import Exp_Basic
        from models import ModernTCN
        Exp_Classification.infer = infer_cls
        Exp = Exp_Classification

    if args.is_training:
        pass
    else:
        ii = 0
        exp = Exp(args)  # set experiments
        setting = '{}_{}_{}_ft{}_sl{}_pl{}_dim{}_nb{}_lk{}_sk{}_ffr{}_ps{}_str{}_multi{}_merged{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.pred_len,
            args.dims[0],
            args.num_blocks[0],
            args.large_size[0],
            args.small_size[0],
            args.ffn_ratio,
            args.patch_size,
            args.patch_stride,
            args.use_multi_scale,
            args.small_kernel_merged,
            args.des,
            ii)



        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.infer(setting, test=1)
        torch.cuda.empty_cache()
