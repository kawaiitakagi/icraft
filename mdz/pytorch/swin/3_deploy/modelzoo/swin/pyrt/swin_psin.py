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
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np
from swin_utils import *
import torch



if __name__ == "__main__":
    # 获取yaml
    Yaml_Path = "../cfg/swin.yaml"
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
    cuda_Mode = cfg["imodel"]["cudamode"]
    eval = cfg["imodel"]["eval"]
    JSON_PATH, RAW_PATH = getJrPath(folderPath,stage,run_sim)

    load_mmu = cfg["imodel"]["mmu"]
    load_speedmode = cfg["imodel"]["speedmode"]
    load_compressFtmp = cfg["imodel"]["compressFtmp"]
    ip = str(cfg["imodel"]["ip"])
    show = cfg["imodel"]["show"]
    save = cfg["imodel"]["save"]

    imgRoot = os.path.abspath(cfg["dataset"]["dir"])
    imgList = os.path.abspath(cfg["dataset"]["list"])
    resRoot = os.path.abspath(cfg["dataset"]["res"])
    if not os.path.exists(resRoot):
        os.makedirs(resRoot)
    else:
        pass
    if eval:
        acc1_meter = AverageMeter()
        acc5_meter = AverageMeter()
        labelList = os.path.abspath(cfg["dataset"]["label"])
        lables = open(labelList).read().splitlines()

    transform = transforms.Compose([
            transforms.Resize(size=256, max_size=None, antialias=None),
            transforms.CenterCrop(size=(224, 224))]
            # transforms.ToTensor(),
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
        )

    # 加载network
    network = loadNetwork(JSON_PATH, RAW_PATH)
  	# 初始化netinfo
    netinfo = Netinfo(network)
    # 选择对网络进行切分
    network_view = network.view(netinfo.inp_shape_opid + 1)
    # 打开device
    device = openDevice(run_sim, ip, netinfo.mmu or load_mmu,cuda_Mode)
    # 初始化session
    session = initSession(run_sim, network_view, device, netinfo.mmu or load_mmu, load_speedmode, load_compressFtmp)
	#开启计时功能
    session.enableTimeProfile(True)
	#session执行前必须进行apply部署操作
    session.apply()
    idx = 0
    for line in tqdm(open(imgList, "r")):
        line = line.strip()
        img_path = os.path.join(imgRoot, line)
        # pre process
        image = Image.open(img_path, mode='r')
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img = np.expand_dims(np.array(transform(image)),0).copy()
        # input_tensor = Tensor(img,Layout("NHWC"))
        input_tensor = numpy2Tensor(img,network)
        # dma init(if use imk)
        dmaInit(run_sim,netinfo.ImageMake_on, netinfo.i_shape[0][1:],input_tensor, device)
        # run
        output_tensors = session.forward([input_tensor])
        out_cls = np.array(output_tensors[0]).astype(np.float32)

        # print(output_tensors[0])
        if not run_sim: 
            device.reset(1)
        if not run_sim and not eval :
            calctime_detail(session,network, name="./"+network.name+"_time.xlsx")

        # post process
        if eval :
            target = torch.Tensor([int(lables[idx])])

            out = torch.Tensor(out_cls)
            acc1, acc5 = accuracy(out, target, topk=(1, 5))

            acc1_meter.update(acc1.item(), target.size(0))
            acc5_meter.update(acc5.item(), target.size(0))
            if idx % 10 == 0:
                print(f'Test: [{idx}/{5000}]\t'
                       f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                       f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                       )
            idx = idx +1
        if show or save:
            print("\n",out_cls.argmax())
            print("The pred class is: ",index2label[out_cls.argmax()])
            output_path = os.path.join(resRoot, line.replace('.jpeg','_result.jpeg').replace('.jpg','_result.jpg'))
            I1 = ImageDraw.Draw(image)
            font = ImageFont.truetype('arial.ttf', 36)  # 替换为你的字体路径和大小
            I1.text((10, 10), index2label[out_cls.argmax()], font=font, fill =(255, 0, 0))
            if show:
                image.show()
            if save:
                print('result save at',output_path)
                image.save(output_path)
    if not run_sim: Device.Close(device)    
    if eval:
        print("The following is precision information")
        print(f'top1:{acc1_meter.avg:.3f}%, top5:{acc5_meter.avg:.3f}%')
        print("Above is precision information")
        with open(network.name+"_metrics.log",'w+') as f:
            f.write(f'top1:{acc1_meter.avg:.3f}%, top5:{acc5_meter.avg:.3f}%')
