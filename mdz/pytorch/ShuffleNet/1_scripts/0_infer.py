import torch 
from PIL import Image
from torchvision import transforms
import torchvision.models as models 

# load pytorch预训练模型并进行单张图片推理
model_urls = {
    'shufflenetv2_x0.5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
    'shufflenetv2_x1.5': None,
    'shufflenetv2_x2.0': None,
}

# preprocess input 
filename = R'./dog.jpg'
input_image = Image.open(filename)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model,[3,224,224]
print('input size =',input_tensor.shape)

# load model 
model = models.shufflenet_v2_x1_0(pretrained=True)

model.eval()
output = model(input_batch)
print('output size =',output[0].shape)
probabilities = torch.nn.functional.softmax(output[0], dim=0)
# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
print('--------Classification results----------')
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())



