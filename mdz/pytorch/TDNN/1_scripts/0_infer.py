import argparse
import functools
import sys
sys.path.append(R"../0_TDNN")
from macls.predict import MAClsPredictor
from macls.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/tdnn.yml',   '配置文件')
add_arg('audio_path',       str,    'dataset/126153-9-0-11.wav', '测试音频路径')
add_arg('model_path',       str,    '../weights/TDNN_Fbank/best_model/', '模型权重文件路径')
args = parser.parse_args()
print_arguments(args=args)
# 备注： 太短的音频(0.3175ms < 0.5ms)会无法预测
# 获取识别器

predictor = MAClsPredictor(configs=args.configs,
                           model_path=args.model_path,
                           use_gpu=False)

label, score = predictor.predict(audio_data=args.audio_path)

print(f'音频：{args.audio_path} 的预测结果标签为：{label}，得分：{score}')
