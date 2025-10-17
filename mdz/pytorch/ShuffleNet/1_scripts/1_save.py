import torch 
import torchvision.models as models 
# 版本检查

RED = '\033[31m'       # 设置前景色为红色
RESET = '\033[0m'      # 重置所有属性到默认值
ver = torch.__version__
assert ("1.6" in ver) or ("1.9" in ver)or ("2.0.1" in ver), f"{RED}Unsupported PyTorch version: {ver}{RESET}"

# 单张图片推理
model_urls = {
    'shufflenetv2_x0.5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
    'shufflenetv2_x1.5': None,
    'shufflenetv2_x2.0': None,
}

# load model 
model = models.shufflenet_v2_x1_0(pretrained=True)
# print(model)
model.eval()
dummy_input = torch.randn(1,3,224,224)
output = model(dummy_input)
pt_path = '../2_compile/fmodel/ShuffleNetv2_224x224.pt'
torch.jit.save(torch.jit.trace(model,dummy_input),pt_path)
print('TorchScript export success, saved as %s' % pt_path)



