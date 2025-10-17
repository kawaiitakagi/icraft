from icraft.xir import * 
from icraft.xrt import * 
from icraft.host_backend import *
from icraft.buyibackend import * 

# import torch
import cv2
import numpy as np
import json

GENERATED_JSON_FILE = "../3_deploy/modelzoo_v3.1.0/yolov8_cls/imodel/16/yolov8s_cls_BY.json"
GENERATED_RAW_FILE = "../3_deploy/modelzoo_v3.1.0/yolov8_cls/imodel/16/yolov8s_cls_BY.raw"

# load model
generated_network = Network.CreateFromJsonFile(GENERATED_JSON_FILE)
generated_network.loadParamsFromFile(GENERATED_RAW_FILE)
print('INFO: Create network!')

# # CV2 preprocessing #
def preprocess_image(image_path):
    img_raw = cv2.imread(image_path)

    # scale by short edge
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
    img = img.transpose(1,2,0) # CHW to HWC
    
    img_ = img.reshape(1,224,224,3) # HWC to NHWC
    img_ = np.ascontiguousarray(img_)
    input_tensor = Tensor(img_, Layout("NHWC"))
    return input_tensor


IMAGE_PATH = "../2_compile/qtset/imagenet/ILSVRC2012_val_00000441.JPEG"
LABEL_PATH = "../2_compile/qtset/imagenet_labels.json"

image = preprocess_image(IMAGE_PATH)

# create forward Session
session = Session.Create([HostBackend],generated_network.view(0),[HostDevice.Default()])
session.apply()
generated_output = session.forward([image])
outputs = []

for i in range(len(generated_output)):
    output = np.array(generated_output[i])
    outputs.append(output[0])
    
# in case your torch version is not recommended
def topk(array, k):
    """Returns the top k values and their indices from a numpy array."""
    array = np.array(array).flatten()
    indices = np.argpartition(array, -k)[-k:]  # Get top k indices
    topk_values = array[indices]
    sorted_indices = np.argsort(-topk_values)  # Sort indices based on values
    topk_indices = indices[sorted_indices]
    topk_values = topk_values[sorted_indices]
    return topk_values, topk_indices

values, indices = topk(outputs, 5)

with open(LABEL_PATH, 'r') as f:
    imagenet_classes = json.load(f)

top_classes = [(imagenet_classes[str(idx.item())], f'{prob.item():.2f}%') for idx, prob in zip(indices, values)]
print(top_classes)

