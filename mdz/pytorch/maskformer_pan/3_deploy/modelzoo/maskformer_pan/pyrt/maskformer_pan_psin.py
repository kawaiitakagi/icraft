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

def panoptic_inference(mask_cls, mask_pred):
    scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
    mask_pred = mask_pred.sigmoid()

    keep = labels.ne(133) & (scores >0.8)
    cur_scores = scores[keep]
    cur_classes = labels[keep]
    cur_masks = mask_pred[keep]
    cur_mask_cls = mask_cls[keep]
    cur_mask_cls = cur_mask_cls[:, :-1]

    cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

    h, w = cur_masks.shape[-2:]
    panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
    segments_info = []

    current_segment_id = 0

    if cur_masks.shape[0] == 0:
        # We didn't detect any mask :(
        return panoptic_seg, segments_info
    else:
        # take argmax
        cur_mask_ids = cur_prob_masks.argmax(0)
        stuff_memory_list = {}
        for k in range(cur_classes.shape[0]):
            pred_class = cur_classes[k].item()
            isthing = pred_class in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                                    19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 
                                    36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53
                                    , 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
                                        72, 73, 74, 75, 76, 77, 78, 79]

            mask = cur_mask_ids == k
            mask_area = mask.sum().item()
            original_area = (cur_masks[k] >= 0.5).sum().item()

            if mask_area > 0 and original_area > 0:
                if mask_area / original_area < 0.8:
                    continue

                # merge stuff regions
                if not isthing:
                    if int(pred_class) in stuff_memory_list.keys():
                        panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                        continue
                    else:
                        stuff_memory_list[int(pred_class)] = current_segment_id + 1

                current_segment_id += 1
                panoptic_seg[mask] = current_segment_id

                segments_info.append(
                    {
                        "id": current_segment_id,
                        "isthing": bool(isthing),
                        "category_id": int(pred_class),
                    }
                )

        return panoptic_seg, segments_info

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
        pass

    # 加载network
    network = loadNetwork(JSON_PATH, RAW_PATH)
  	# 初始化netinfo
    netinfo = Netinfo(network)




    def get_transform_fix_size(self, image):
        h, w = image.shape[:2]
        if self.is_range:
            size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
        else:
            size = np.random.choice(self.short_edge_length)

        newh, neww = (netinfo.i_shape[0][1],netinfo.i_shape[0][2]) # revised
        return ResizeTransform(h, w, newh, neww, self.interp)
    from detectron2.data.transforms.augmentation_impl import ResizeShortestEdge
    ResizeShortestEdge.get_transform = get_transform_fix_size

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
    idx = 0
    if eval :
        from maskformer_pan_val import *

        eval_res = main(session,run_sim,netinfo,network,device,imgRoot)
        pass
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
            mask_pred_result = sem_seg_postprocess(mask_pred_result, image.image_sizes[0], height, width)
            panoptic_r = panoptic_inference(mask_cls_result, mask_pred_result)

            metadata = MetadataCatalog.get('coco_2017_val_panoptic')
            
            visualizer = Visualizer(original_image, metadata, instance_mode=ColorMode.IMAGE)
            panoptic_seg, segments_info = panoptic_r

            vis_output = visualizer.draw_panoptic_seg_predictions(panoptic_seg, segments_info)
            output_path = os.path.join(resRoot, line.replace('.jpeg','_result.jpeg').replace('.jpg','_result.jpg'))
            if show:
                cv2.imshow("maskformer_pan res", vis_output.get_image()[:, :, ::-1])
                cv2.waitKey(0)
            if save:
                print('result save at',output_path)
                vis_output.save(output_path)
    if not run_sim: Device.Close(device)    
    if eval:
        pq = eval_res["panoptic_seg"]["PQ"]
        print("The following is precision information")
        print(f'PQ:{pq:.3f}')
        print("Above is precision information")

        with open(network.name+"_metrics.log",'w+') as f:
            f.write(f'PQ:{pq}%')
