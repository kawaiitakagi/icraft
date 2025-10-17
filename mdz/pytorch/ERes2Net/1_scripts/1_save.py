import argparse
import functools
import sys
sys.path.append(R"../0_ERes2Net")
import torch 

from mvector.predict import MVectorPredictor
from mvector.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/eres2net.yml',        '配置文件')
add_arg('audio_path',       str,    'dataset/a_1.wav',          '预测第一个音频')
add_arg('model_path',       str,    '../weights/ERes2Net_TSTP_Fbank/best_model/', '模型权重文件路径')
add_arg('dst_path',         str,    '../2_compile/fmodel/ERes2Net_predictor_1x365x80.onnx', '导出的onnx模型文件路径')
args = parser.parse_args()
print_arguments(args=args)

# 获取识别器
model = MVectorPredictor(configs=args.configs,
                             model_path=args.model_path,
                             use_gpu=False)
# 加载音频文件1，并进行预处理
input_data = model._load_audio(audio_data=args.audio_path, sample_rate=16000)
input_data = torch.tensor(input_data.samples, dtype=torch.float32).unsqueeze(0)
print('input_data_=',input_data.shape,input_data)
audio_feature = model._audio_featurizer(input_data)
print('audio_feature =',audio_feature.shape)
# 执行预测
output = model.predictor(audio_feature)
print('output =',output.shape)
# 导出ONNX模型
input_names = ["audio_feature"]
output_names = ["logits"]
torch.onnx.export(
    model.predictor,
    audio_feature,
    args.dst_path,
    verbose=False,
    keep_initializers_as_inputs=False,
    opset_version=17,
    input_names=input_names,
    output_names=output_names,)
print(f'ONNX export success, saved in: {args.dst_path} ')

