import argparse
import functools
import torch 
import sys
sys.path.append(R"../0_ResNetSE")
from macls.predict import MAClsPredictor
from macls.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/resnet_se.yml',   '配置文件')
add_arg('audio_path',       str,    'dataset/126153-9-0-11.wav', '测试音频路径')
add_arg('model_path',       str,    '../weights/ResNetSE_Fbank/best_model/', '模型权重文件路径')
add_arg('dst_path',       str,    '../2_compile/fmodel/ResNetSE_predictor_1x398x80.pt', '导出的pt模型文件路径')
args = parser.parse_args()
print_arguments(args=args)
# 备注： 太短的音频(0.3175ms < 0.5ms)会无法预测
# 获取识别器

model = MAClsPredictor(configs=args.configs,
                           model_path=args.model_path,
                           use_gpu=False)
# 加载音频文件，并进行预处理
input_data = model._load_audio(audio_data=args.audio_path, sample_rate=16000)
input_data = torch.tensor(input_data.samples, dtype=torch.float32).unsqueeze(0)
audio_feature = model._audio_featurizer(input_data)
print('audio_feature =',audio_feature.shape)
# 执行预测
output = model.predictor(audio_feature)
print('output =',output.shape)
# 导出模型
pt_model = model.predictor
traced_model = torch.jit.trace(pt_model,audio_feature,strict=False)
torch.jit.save(traced_model,args.dst_path)
print(f'TorchScript export success, saved in: {args.dst_path} ')
