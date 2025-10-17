WEIGHTS_PATH = "../2_compile/fmodel/yolov8s-cls-224x224.pt"
IMAGE_PATH = "../2_compile/qtset/imagenet/ILSVRC2012_val_00000441.JPEG"
LABEL_PATH = "../2_compile/qtset/imagenet_labels.json"

import cv2
import torch
import numpy as np
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################################################################
# # Simple preprocessing #
# def preprocess_image(image_path):
#     img_raw = cv2.imread(image_path)
#     img_resize = cv2.resize(img_raw, (224, 224))  
#     img = img_resize.transpose((2, 0, 1))[::-1]   # HWC to CHW and BGR to RGB
#     # img = img_resize.transpose((0, 1, 2))[::-1]   
#     img = np.ascontiguousarray(img)
#     img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(device) / 255
#     return img_tensor
############################################################################
# # PIL preprocessing #
# from torchvision import transforms
# from PIL import Image
# def preprocess_image(image_path):
#     transform = transforms.Compose([
#     transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR, max_size=None, antialias=Warning),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])
#     ])
#     img_raw = cv2.imread(image_path)
#     img = transform(Image.fromarray(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)))
#     im = img.float().unsqueeze(0)
#     return im
############################################################################
# # CV2 preprocessing #
def preprocess_image(image_path):
    img_raw = cv2.imread(image_path)

    #scale by short edge
    h, w = img_raw.shape[:2]
    ratio_h = 224 / h
    ratio_w = 224 / w
    scale = np.max([ratio_h, ratio_w])
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img_resize = cv2.resize(img_raw, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # center crop
    start_w = int((new_w - 224) / 2.0)
    start_h = int((new_h - 224) / 2.0)
    cropped_img = img_resize[start_h:start_h + 224, start_w:start_w + 224].copy()

    img = cropped_img.transpose((2, 0, 1))[::-1]   # HWC to CHW and BGR to RGB
    img = np.ascontiguousarray(img)
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device) / 255
    return img_tensor
############################################################################

image = preprocess_image(IMAGE_PATH)
model = torch.jit.load(WEIGHTS_PATH).to(device)
model.eval()
output = model(image)
print(output)

values, indices = torch.topk(output, 5)

with open(LABEL_PATH, 'r') as f:
    imagenet_classes = json.load(f)

top_classes = [(imagenet_classes[str(idx.item())], f'{prob.item():.2f}%') for idx, prob in zip(indices[0], values[0])]
print(top_classes)
