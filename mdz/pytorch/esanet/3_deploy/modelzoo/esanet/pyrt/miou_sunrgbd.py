import sys
sys.path.append(R"../../../Deps/modelzoo")
import icraft
from icraft.xir import *
from icraft.xrt import *
from icraft.buyibackend import *
from icraft.host_backend import *
from pyrtutils.icraft_utils import *
from pyrtutils.Netinfo import Netinfo
from pyrtutils.utils import *
from pyrtutils.et_device import *
from pyrtutils.calctime_utils import *

from tqdm import tqdm
import yaml
import os
import cv2
import numpy as np
import torch.nn.functional as F

from src.prepare_data import prepare_data
import torch
import matplotlib.pyplot as plt

def _load_img(fp):
    img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def compute_miou(predictions, labels):
    predictions = predictions.astype(np.int32)
    labels = labels.astype(np.int32)

    labels[labels == 255] = 0  # 假设255是无效的像素
    predictions[predictions == 255] = 0  # 同样处理预测
    unique_labels = np.unique(labels)
    unique_predictions = np.unique(predictions)
    num_classes = len(unique_labels)  # 动态确定类别数

    # print(f"Unique labels: {unique_labels}")
    # print(f"Unique predictions: {unique_predictions}")
    
    # 初始化交集和并集
    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)

    # 计算每个类别的交集和并集
    for cls in range(num_classes):
        # 找到当前类别的预测和真实值
        pred_cls = (predictions == cls)
        label_cls = (labels == cls)

        # 计算交集和并集
        intersection[cls] = np.sum(pred_cls & label_cls)
        union[cls] = np.sum(pred_cls | label_cls)

    # 计算每个类别的 IoU
    iou = intersection / (union + 1e-6)  # 添加一个小值以避免除以零
    valid_iou = iou[union > 1e-5]
    # valid_iou = iou[iou >1e-5]
    miou = np.mean(valid_iou) if valid_iou.size > 0 else 0  # 计算有效类别的 mIoU
    return miou, iou


if __name__ == "__main__":
    # parser = ArgumentParserRGBDSegmentation(
    #     description='Efficient RGBD Indoor Sematic Segmentation (Inference)',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.set_common_args()
    # parser.add_argument('--depth_scale', type=float,
    #                     default=1.0,
    #                     help='Additional depth scaling factor to apply.')
    # args = parser.parse_args()

    # dataset
    # args.pretrained_on_imagenet = False  # we are loading other weights anyway
    dataset, preprocessor = prepare_data(dataset='sunrgbd', with_input_orig=True)
    n_classes = dataset.n_classes_without_void


    # 获取yaml
    Yaml_Path = "../cfg/esanet_test.yaml"
    if len(sys.argv) < 2:
        mprint("Info:未传入yaml参数,读入默认yaml文件:‌{}进行相关配置.".format(Yaml_Path), VERBOSE, 0)        
    if len(sys.argv) == 2:
        Yaml_Path = sys.argv[1]
        mprint("info:传入yaml文件:‌{}进行相关配置.".format(Yaml_Path), VERBOSE, 0)        
    if len(sys.argv) > 2:
        mprint("info:传入参数数量错误,请检查运行命令!", VERBOSE, 0)        
        sys.exit(1)
    # 从yaml里读入配置
    cfg = yaml.load(open(Yaml_Path, "r"), Loader=yaml.FullLoader)   
    folderPath = cfg["imodel"]["dir"]
    stage = cfg["imodel"]["stage"]
    run_sim = cfg["imodel"]["sim"]
    JSON_PATH, RAW_PATH = getJrPath(folderPath,stage,run_sim)

    load_mmu = cfg["imodel"]["mmu"]
    load_speedmode = cfg["imodel"]["speedmode"]
    load_compressFtmp = cfg["imodel"]["compressFtmp"]
    ip = str(cfg["imodel"]["ip"])
    save = cfg["imodel"]["save"]
    show = cfg["imodel"]["show"]

    imgRoot = os.path.abspath(cfg["dataset"]["dir"])
    imgList = os.path.abspath(cfg["dataset"]["list"])
    resRoot = cfg["dataset"]["res"]

    # 加载network
    network = loadNetwork(JSON_PATH, RAW_PATH)
  	# 初始化netinfo
    netinfo = Netinfo(network)
    # 选择对网络进行切分
    network_view = network.view(netinfo.inp_shape_opid + 1)
    # 打开device
    device = openDevice(run_sim, ip, netinfo.mmu or load_mmu)
    # 初始化session
    session = initSession(run_sim, network_view, device, netinfo.mmu or load_mmu, load_speedmode, load_compressFtmp)
	#开启计时功能
    session.enableTimeProfile(True)
	#session执行前必须进行apply部署操作
    session.apply()


    x = 0
    n = 0
    for line in tqdm(open(imgList, "r")):
        line1 = line.strip().split(';')[0]
        line2 = line.strip().split(';')[-1]
        # img_path = os.path.join(imgRoot, line1)
        img_path = imgRoot + '/' + 'RGB/' + line1
        # depth_path = os.path.join(imgRoot, line2)
        depth_path = imgRoot + '/' + 'Depth/' + line2
        # pre process
        img_rgb = _load_img(img_path)
        img_depth = _load_img(depth_path).astype('float32')
        h, w, _ = img_rgb.shape

        # preprocess sample
        sample = preprocessor({'image': img_rgb, 'depth': img_depth})
        # add batch axis and copy to device
        image = sample['image'][None]
        depth = sample['depth'][None]

        input_0 =  np.transpose(np.array(image).astype(np.float32), (0, 2, 3, 1))
        input_1 =  np.transpose(np.array(depth).astype(np.float32), (0, 2, 3, 1))

        input_tensor_0 = numpy2Tensor(input_0, network.ops[0].outputs[0])
        input_tensor_1 = numpy2Tensor(input_1, network.ops[0].outputs[1])
    
        # dma init(if use imk)
        dmaInit(run_sim,netinfo.ImageMake_on, netinfo.i_shape[0][1:],(input_tensor_0,input_tensor_1), device)

        # run
        output_tensors = session.forward([input_tensor_0,input_tensor_1])

        if not run_sim: 
            device.reset(1)
            # calctime_detail(session,network, name="./"+network.name+"_time.xlsx")

        # post process
        pred = np.array(output_tensors[0]).astype(np.float32)
        pred = np.transpose(pred, (0, 3, 1, 2))
        pred = torch.from_numpy(pred)
        pred = F.interpolate(pred, (h, w),
                             mode='bilinear', align_corners=False)
        pred = torch.argmax(pred, dim=1)
        pred = pred.cpu().numpy().squeeze().astype(np.uint8)
        # # show result
        # pred_colored = dataset.color_label(pred, with_void=False)
        # fig, axs = plt.subplots(1, 3, figsize=(16, 3))
        # [ax.set_axis_off() for ax in axs.ravel()]
        # axs[0].imshow(img_rgb)
        # axs[1].imshow(img_depth, cmap='gray')
        # axs[2].imshow(pred_colored)

        # plt.suptitle(f"Image: ({os.path.basename(img_path)}, "
        #              f"{os.path.basename(depth_path)}), Model: icraft")
        # plt.show()

        # compute iou 
        label_path = depth_path.replace('Depth','labels')
        # print(label_path)
        label_orig = cv2.imread(label_path,-1)
        # ignore void pixels
        mask = label_orig > 0
        label = label_orig[mask]
        pred = pred[mask]

        # # In the label 0 is void but in the prediction 0 is wall.
        # # In order for the label and prediction indices to match
        # # we need to subtract 1 of the label.
        label -= 1
        # label = label_orig - 1

        miou, iou_per_class = compute_miou(label, pred)

        print(f"Mean IoU: {miou:.4f}")
        x = x + miou
        n = n+1
    print('last result of 2000 images:', x/n)
    miou_value = x/n
    log_file_name = './esanet_metrics.log'
    with open(log_file_name,"w") as log:
        log.write(f"miou: {miou_value * 100}%\n")

    if not run_sim: Device.Close(device)    
