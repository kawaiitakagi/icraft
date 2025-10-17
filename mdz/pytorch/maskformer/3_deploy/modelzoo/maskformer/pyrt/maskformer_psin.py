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
import torch
from torch.nn import functional as F
from detectron2.data.transforms.transform  import ResizeTransform
from detectron2.data import MetadataCatalog
from detectron2.structures.image_list import ImageList
from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.modeling.postprocessing import sem_seg_postprocess

if __name__ == "__main__":
    # 获取yaml
    Yaml_Path = "../cfg/maskformer_test.yaml"
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
    # 加载network
    network = loadNetwork(JSON_PATH, RAW_PATH)
  	# 初始化netinfo
    netinfo = Netinfo(network)

    def get_transform_fix_size(self, image):
        h, w = image.shape[:2]
        newh, neww = (netinfo.i_shape[0][1],netinfo.i_shape[0][2]) # revised
        return ResizeTransform(h, w, newh, neww, self.interp)
    from detectron2.data.transforms.augmentation_impl import ResizeShortestEdge
    ResizeShortestEdge.get_transform = get_transform_fix_size

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
    if eval :
        from maskformer_val import *

        eval_res = main(session,run_sim,netinfo,network,device,imgRoot)
    else:
        for line in tqdm(open(imgList, "r")):
            line = line.strip()
            img_path = os.path.join(imgRoot, line)
            # pre process
            image = read_image(img_path, format="BGR")
            original_image = image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = ResizeShortestEdge(short_edge_length=[netinfo.i_shape[0][1], netinfo.i_shape[0][2]], max_size=2048).get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            # pixel_mean = torch.tensor([[[123.6750]],[[116.2800]],[[103.5300]]])
            # pixel_std = torch.tensor([[[58.3950]],[[57.1200]],[[57.3750]]])
            image = ImageList.from_tensors([image], 32)
            # image.tensor = (image.tensor - pixel_mean) / pixel_std
            # img = np.expand_dims(image,0).copy()
            # input_tensor = Tensor(img,Layout("NHWC"))
            input_tensor = numpy2Tensor(image.tensor.numpy().transpose(0 ,2 ,3 ,1).copy(),network)
            # dma init(if use imk)
            dmaInit(run_sim,netinfo.ImageMake_on, netinfo.i_shape[0][1:],input_tensor, device)
            # run
            output_tensors = session.forward([input_tensor])

            if not run_sim: 
                device.reset(1)
            if not run_sim and not eval :
                calctime_detail(session,network, name="./"+network.name+"_time.xlsx")

            out_0 = np.array(output_tensors[0]).astype(np.float32)
            out_1 = np.array(output_tensors[1]).astype(np.float32)
            mask_cls_result = torch.Tensor(np.array(out_0))[0]
            mask_pred_result = torch.Tensor(np.array(out_1))[0].permute(2, 0, 1)
            mask_cls = F.softmax(mask_cls_result, dim=-1)[..., :-1]
            mask_pred = mask_pred_result.sigmoid()
            r = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            r = sem_seg_postprocess(r, image.image_sizes[0], height, width)
            metadata = MetadataCatalog.get('ade20k_sem_seg_val' if len(('ade20k_sem_seg_val',)) else "__unused")
            visualizer = Visualizer(original_image, metadata, instance_mode=ColorMode.IMAGE)
            vis_output = visualizer.draw_sem_seg(r.argmax(dim=0))

            # print(output_tensors[0])
            # print(output_tensors[1])
            # post process

            output_path = os.path.join(resRoot, line.replace('.jpeg','_result.jpeg').replace('.jpg','_result.jpg'))
  
            if show:
                cv2.imshow("maskformer res", vis_output.get_image()[:, :, ::-1])
                cv2.waitKey(0)
            if save:
                print('result save at',output_path)
                vis_output.save(output_path)
    if not run_sim: Device.Close(device)    
    if eval:
        miou = eval_res['sem_seg']['mIoU']
        print("The following is precision information")
        print(f'mIou:{miou:.3f}')
        print("Above is precision information")

        with open(network.name+"_metrics.log",'w+') as f:
            f.write(f'mIou:{miou}%')

