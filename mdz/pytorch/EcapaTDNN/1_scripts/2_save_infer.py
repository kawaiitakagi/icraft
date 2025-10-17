import torch 
import onnx
import onnxruntime
import numpy as np
import sys
sys.path.append(R"../0_EcapaTDNN")
import argparse
import functools

from macls.predict import MAClsPredictor
from macls.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

add_arg('configs',          str,    'configs/ecapa_tdnn.yml',   '配置文件')
add_arg('audio_path',       str,    'dataset/126153-9-0-11.wav', '音频路径')
add_arg('model_path',       str,    '../weights/EcapaTdnn_Fbank/best_model/', '导出的预测模型文件路径')
add_arg('onnx_path',          str,    '../2_compile/fmodel/EcapaTDNN_predictor_1x398x80.onnx', '导出的onnx模型文件路径')
args = parser.parse_args()
print_arguments(args=args)

# 获取识别器

predictor = MAClsPredictor(configs=args.configs,
                           model_path=args.model_path,
                           use_gpu=False)


# 加载音频文件，并进行预处理
input_data = predictor._load_audio(audio_data=args.audio_path, sample_rate=16000)
input_data = torch.tensor(input_data.samples, dtype=torch.float32).unsqueeze(0)
print('input_data =',input_data.shape)
# 提取音频特征
audio_feature = predictor._audio_featurizer(input_data).to(torch.device("cpu"))
print('audio_feature =',audio_feature.shape)
# 加载并验证onnx模型有效性
onnx_model = onnx.load(args.onnx_path)
onnx.checker.check_model(onnx_model)
# 创建推理会话
ort_session = onnxruntime.InferenceSession(args.onnx_path,providers=["CPUExecutionProvider"])
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
# 使用ONNX Runtime进行推理
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(audio_feature)}
ort_outs = ort_session.run(None, ort_inputs) # lists of ndarray

output = torch.tensor(ort_outs)[0]
result = torch.nn.functional.softmax(output, dim=-1)[0]
result = result.data.cpu().numpy()
# 最大概率的label
lab = np.argsort(result)[-1]
score = result[lab]
CLASS_LABEL = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
label = CLASS_LABEL[lab]
score = round(float(score), 5)
print(f'音频：{args.audio_path} 的预测结果标签为：{label}，得分：{score}')


# # load pt model
# model = torch.jit.load(args.onnx_path)
# # 执行预测
# output = model(audio_feature)
# print('output =',output,output.shape)
# result = torch.nn.functional.softmax(output, dim=-1)[0]
# result = result.data.cpu().numpy()
# 最大概率的label
# lab = np.argsort(result)[-1]
# score = result[lab]
# CLASS_LABEL = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
# label = CLASS_LABEL[lab]
# score = round(float(score), 5)
# print(f'音频：{args.audio_path} 的预测结果标签为：{label}，得分：{score}')
