# 安装
## Stable Baselines3 安装
```
pip install stable-baselines3
```
默认安装下gymnasium包组件缺失。补充安装
```
pip install gymnasium[box2d]
pip install gymnasium[other]
```
## Baselines3 Zoo 安装
From source
```
cd 0_sac_donkey_car
pip install -e .

```
## onnx 安装
```
pip install onnx
pip install onnxruntime
```
# donkey_car 仿真环境安装
## 1、https://github.com/tawnkramer/gym-donkeycar/releases，下载并启动donkey_car仿真器
## 2、wrapper安装

```
cd 0_sac_donkey_car\aae-train-donkeycar-master
pip install -e .
```
## 3、gym_donkey安装
```
cd 0_sac_donkey_car\gym-donkeycar-22.11.06
pip install -e .
```

# icraft python runtime安装
> 对于该模型，我们将其划分为特征提取和决策两个部分，若希望将特征提取部分运行在torch框架上，请修改weights\models\sac\donkey-mountain-track-v0_1\donkey-mountain-track-v0\
> 下的config.yml的第12行，此时将使用cpu进行浮点计算

# 运行
## 0_infer
```python
cd 1_scripts
python 0_infer.py 
```

## 1_save
```python
# 请提前下载qtset
cd 1_scripts
python 1_save.py 
```

## 2_save_infer
```python
cd 1_scripts
python 2_save_infer.py 
```