from icraft.xir import *
from icraft.xrt import *
from icraft.buyibackend import *
import torch
import torchvision.transforms as T
from icraft.host_backend import *
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt


def dmaInit(device, input_tensor, shape, imk):
    if imk:
        h,w,c=shape[0],shape[1],shape[2]
        demo_reg_base = 0x1000C0000
        uregion_=device.getMemRegion("udma")
        utensor = input_tensor.to(uregion_)#data transfer ps->udma + IMK(udma->pl)
        ImageMakeRddrBase = utensor.data().addr()
        ImageMakeRlen = ((w * h - 1) // (24 // c) + 1) * 3
        ImageMakeLastSft = w * h - (ImageMakeRlen - 3) // 3 * (24 // c)
        device.defaultRegRegion().write(demo_reg_base + 0x4, ImageMakeRddrBase, True)
        device.defaultRegRegion().write(demo_reg_base + 0x8, ImageMakeRlen, True)
        device.defaultRegRegion().write(demo_reg_base + 0xC, ImageMakeLastSft, True)
        device.defaultRegRegion().write(demo_reg_base + 0x10, c, True)
        device.defaultRegRegion().write(demo_reg_base + 0x1C, 1, True)
        device.defaultRegRegion().write(demo_reg_base + 0x20, 0, True)
        # imk start
        device.defaultRegRegion().write(demo_reg_base, 1, True)
    return 0

def fpgaOPlist(network):
    # only used in adapt&by stage
    customop_set = set()
    oplist = network.ops
    for op in oplist:
        if "customop" in op.typeKey():
            customop_set.add(op.typeKey())
    return customop_set

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]
transform = T.Compose([
    T.Resize([640,640]),

])
def dmaInit(shape,input_tensor,device ):
    h,w,c=shape[0],shape[1],shape[2]
    demo_reg_base = 0x1000C0000
    uregion_=device.getMemRegion("udma")
    utensor = input_tensor.to(uregion_)#data transfer ps->udma + IMK(udma->pl)
    ImageMakeRddrBase = utensor.data().addr()
    ImageMakeRlen = ((w * h - 1) // (24 // c) + 1) * 3
    ImageMakeLastSft = w * h - (ImageMakeRlen - 3) // 3 * (24 // c)
    device.defaultRegRegion().write(demo_reg_base + 0x4, ImageMakeRddrBase, True)
    device.defaultRegRegion().write(demo_reg_base + 0x8, ImageMakeRlen, True)
    device.defaultRegRegion().write(demo_reg_base + 0xC, ImageMakeLastSft, True)
    device.defaultRegRegion().write(demo_reg_base + 0x10, c, True)
    device.defaultRegRegion().write(demo_reg_base + 0x1C, 1, True)
    device.defaultRegRegion().write(demo_reg_base + 0x20, 0, True)
    # imk start
    device.defaultRegRegion().write(demo_reg_base, 1, True)
    return 0

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)
def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()



run_parse = False
JSON_FILE = '../3_deploy/modelzoo/detr/imodel/8/detr_640x640_optimized.json' 
RAW_FILE = '../3_deploy/modelzoo/detr/imodel/8/detr_640x640_optimized.raw'
IMG_PATH = '../2_compile/qtset/detr/000000000632.jpg'

network = Network.CreateFromJsonFile(JSON_FILE)
network.loadParamsFromFile(RAW_FILE)
customop_set=fpgaOPlist(network)
print(customop_set)
IMK = True if "customop::ImageMakeNode" in customop_set else False
print('IMK =',IMK)
device = HostDevice.Default()

sess = Session.Create([ HostBackend ], network.view(0), [HostDevice.Default()])


im = Image.open(IMG_PATH)
img = np.array(transform(im)).reshape(1,640,640,3).astype(np.float32).copy()
# img2 = img2.permute(0, 2, 3, 1).cpu().detach().numpy().astype(np.float32).copy()
if IMK:
    img= img.astype(np.uint8)
else:
    img= img.astype(np.float32)

tensor = Tensor(img, Layout("NHWC"))

sess.enableTimeProfile(True)

sess.apply()

output_tensors = sess.forward([tensor])

# print(output_tensors[0])
# print(output_tensors[1])
device.reset(1)
result = sess.timeProfileResults()
#[总时间，传输时间，硬件时间，余下时间]
# time = np.array(list(result.values()))
# total_hardtime = np.sum(time[:,2])
# print("Total Time: ",np.sum(time[:,0]),"ms")
# print("Travel Time: ",np.sum(time[:,1]),"ms")
# print("Hard Time: ",np.sum(time[:,2]),"ms")
# print("Res Time: ",np.sum(time[:,3]),"ms")

outputs_class_ = np.reshape(output_tensors[0], (1,100,92))
outputs_class_icraft = torch.tensor(outputs_class_).reshape(1,100,92).contiguous()
outputs_coord_ = np.reshape(output_tensors[1], (1,100,4))
outputs_coord_icraft = torch.tensor(outputs_coord_).reshape(1,100,4).contiguous()
outputs_class = outputs_class_icraft
outputs_coord = outputs_coord_icraft
probas = outputs_class.softmax(-1)[0, :, :-1] 
keep = probas.max(-1).values > 0.7

# convert boxes from [0; 1] to image scales
bboxes_scaled = rescale_bboxes(outputs_coord[0, keep], (640, 483))

plot_results(im, probas[keep], bboxes_scaled)

