import argparse
import functools
import torch 
import onnx
import onnxruntime
import numpy as np
import sys
sys.path.append(R"../0_ERes2Net") 
from mvector.predict import MVectorPredictor
from mvector.utils.utils import add_arguments, print_arguments
def feature_refiner(ori_audio_feature,max_freq_length=365,freq_size=80):
    pad_features = torch.zeros((1, max_freq_length, freq_size), dtype=torch.float32)
    if ori_audio_feature.shape[1]  < max_freq_length:#pad audio_feature
                pad_features[:, :ori_audio_feature.shape[1], :] = ori_audio_feature 
    else:# audio_feature.shape[1] >  398,crop audio_feature
        pad_features = ori_audio_feature[:,:max_freq_length,:]
    return pad_features
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# add_arg('configs',          str,    'configs/eres2net.yml',        '配置文件')
add_arg('configs',          str,    'configs/tdnn.yml',        '配置文件')
add_arg('audio_path1',      str,    'dataset/a_1.wav',          '预测第一个音频')
add_arg('audio_path2',      str,    'dataset/b_2.wav',          '预测第二个音频')
add_arg('threshold',        float,  0.7,                        '判断是否为同一个人的阈值')
add_arg('model_path',       str,    '../weights/TDNN_Fbank/best_model/', '导出的预测模型文件路径')
add_arg('onnx_path',          str,    '../2_compile/fmodel/TDNN_predictor_1x365x80.onnx', '导出的onnx模型文件路径')
args = parser.parse_args()
print_arguments(args=args)

# 获取识别器
predictor = MVectorPredictor(configs=args.configs,
                             model_path=args.model_path,
                             use_gpu=False)
# 加载音频文件1，并进行预处理
input_data_1 = predictor._load_audio(audio_data=args.audio_path1, sample_rate=16000)
input_data_1 = torch.tensor(input_data_1.samples, dtype=torch.float32).unsqueeze(0)
print('input_data_1 =',input_data_1.shape,input_data_1)
audio_feature_1 = predictor._audio_featurizer(input_data_1)
print('audio_feature_1 =',audio_feature_1.shape)
# 加载音频文件2，并进行预处理
input_data_2 = predictor._load_audio(audio_data=args.audio_path2, sample_rate=16000)
input_data_2 = torch.tensor(input_data_2.samples, dtype=torch.float32).unsqueeze(0)
print('input_data_2 =',input_data_2.shape,input_data_2)
audio_feature_2 = predictor._audio_featurizer(input_data_2)
print('audio_feature_2 =',audio_feature_2.shape)
#将音频特征统一补齐或裁剪至max_freq_length
max_freq_length = 365
freq_size = 80
audio_feature_1 = feature_refiner(audio_feature_1,max_freq_length,freq_size)
audio_feature_2 = feature_refiner(audio_feature_2,max_freq_length,freq_size)
# 加载并验证onnx模型有效性
onnx_model = onnx.load(args.onnx_path)
onnx.checker.check_model(onnx_model)
print('ONNX model check done!')
# 创建推理会话
ort_session = onnxruntime.InferenceSession(args.onnx_path,providers=["CPUExecutionProvider"])

# 使用ONNX Runtime执行音频文件1的推理
ort_inputs_1 = {ort_session.get_inputs()[0].name: to_numpy(audio_feature_1)}
ort_outs_1 = ort_session.run(None, ort_inputs_1) # lists of ndarray
output_1 = torch.tensor(ort_outs_1)[0][0]
print('output_1=',output_1.shape)
# 使用ONNX Runtime执行音频文件2的推理
ort_inputs_2 = {ort_session.get_inputs()[0].name: to_numpy(audio_feature_2)}
ort_outs_2 = ort_session.run(None, ort_inputs_2) # lists of ndarray
output_2 = torch.tensor(ort_outs_2)[0][0]
print('output_2=',output_2.shape)
# 对角余弦值
dist = np.dot(output_1, output_2) / (np.linalg.norm(output_1) * np.linalg.norm(output_2))
print('dist =',dist)

if dist > args.threshold:
    print(f"{args.audio_path1} 和 {args.audio_path2} 为同一个人，相似度为：{dist}")
else:
    print(f"{args.audio_path1} 和 {args.audio_path2} 不是同一个人，相似度为：{dist}")

