import torch 
from PIL import Image
from torchvision import transforms
import torchvision.models as models 

# load torchcript模型并进行单张图片推理

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
pt_path = '../2_compile/fmodel/ShuffleNetv2_224x224.pt'
model = torch.load(pt_path,map_location='cpu')
print('Model Load Done')

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



