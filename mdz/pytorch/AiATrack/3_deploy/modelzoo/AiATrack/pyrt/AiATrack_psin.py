import torch
import torch.nn.functional as F
import sys
sys.path.append(R"../../../Deps/modelzoo")
from icraft.xir import *
from icraft.xrt import *
from icraft.host_backend import *
from icraft.buyibackend import *
import numpy as np
import cv2
import os
import yaml
# from datetime import timedelta
from tqdm import tqdm
from process_aiatrack import *
from pyrtutils.icraft_utils import *
from pyrtutils.Netinfo import Netinfo
from pyrtutils.utils import *
from pyrtutils.et_device import *
from pyrtutils.calctime_utils import *


if __name__ == "__main__":
    # 获取yaml
    Yaml_Path = "../cfg/AiATrack.yaml"
    if len(sys.argv) < 2:
        print("Info:未传入yaml参数,读入默认yaml文件:‌{}进行相关配置.".format(Yaml_Path), VERBOSE, 0)        
    if len(sys.argv) == 2:
        Yaml_Path = sys.argv[1]
        print("info:传入yaml文件:‌{}进行相关配置.".format(Yaml_Path), VERBOSE, 0)        
    if len(sys.argv) > 2:
        print("info:传入参数数量错误,请检查运行命令!", VERBOSE, 0)        
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
    calctime = cfg["imodel"]["calctime"]

    imgRoot = os.path.abspath(cfg["dataset"]["dir"])
    imgList = os.path.abspath(cfg["dataset"]["list"])
    resRoot = cfg["dataset"]["res"]

    # 初始化模型输入
    init_ref_mem0 = np.ascontiguousarray(np.random.randn(400, 1, 256)).astype(np.float32)
    init_ref_mem1 = np.ascontiguousarray(np.random.randn(1600, 1, 256)).astype(np.float32)
    init_ref_emb0 = np.ascontiguousarray(np.random.randn(400, 1, 256)).astype(np.float32)
    init_ref_emb1 = np.ascontiguousarray(np.random.randn(1600, 1, 256)).astype(np.float32)
    init_ref_pos0 = np.ascontiguousarray(np.random.randn(400, 4, 64)).astype(np.float32)
    init_ref_pos1 = np.ascontiguousarray(np.random.randn(400, 16, 64)).astype(np.float32)
    embed_bank = torch.tensor(np.fromfile(f"../../../../2_compile/fmodel/embed_bank_1_2_256.ftmp", dtype=np.float32).reshape(1, 2, 256))
    # 归一化系数
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    # 保存txt的路径
    save_dir = resRoot + "res_txt/"
    # 保存img的路径
    save_img_dir = resRoot + "res_img/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # 加载network
    network = loadNetwork(JSON_PATH, RAW_PATH)
  	# 初始化netinfo
    netinfo_net = Netinfo(network)
    # 选择对网络进行切分
    network_view = network.view(0)
    # 打开device
    device = openDevice(run_sim, ip, netinfo_net.mmu or load_mmu)
    # 初始化session
    session = initSession(run_sim, network_view, device, netinfo_net.mmu or load_mmu, load_speedmode, load_compressFtmp)
	#开启计时功能
    session.enableTimeProfile(True)
	#session执行前必须进行apply部署操作
    session.apply()

    # 加载数据集
    TEST_SET = read_list_from_txt(imgList)

    #================逐序列测试================#
    for test_seq in TEST_SET[:]:
        # ============step0:准备测试集=============#
        print('=====Step 0: Prepare test=====')
        print("test_seq = ",test_seq)
        seq_cls = test_seq.split('-')[0]
        # 原LASOT数据集的测试集
        # seq_dir = imgRoot +f'/{seq_cls}/{test_seq}/'
        # gt_file = imgRoot +f'/{seq_cls}/{test_seq}/groundtruth.txt'
        # 内部LASOT_sub62试用以下格式：
        seq_dir = imgRoot +f'/{test_seq}/'
        gt_file = imgRoot +f'/{test_seq}/groundtruth.txt'
        img_dir = seq_dir + 'img/'
        print(f'loading groundtruth from {gt_file}')
        file_list = os.listdir(img_dir)
        ims = sorted(file_list, key=numeric_sort_key)
        tracking_result = []
        raw_result = []
        # 测试结果保存路径
        resfn = f"{save_dir}/{test_seq}.txt"
        resffn = f"{save_dir}/{test_seq}_float.txt"
        rawfn = f"{save_dir}/{test_seq}_raw_bbox.txt"
        print(f"{len(ims)} frames in total.")

        print('=====Step 1: Initialize state and reference cache=====')
        print(f'load initial state from {gt_file}')
        annos = np.loadtxt(gt_file, delimiter=',', dtype=np.float32)
        state = list(annos[0]) 
        # initial state
        ref_mem0 = init_ref_mem0
        ref_mem1 = init_ref_mem1
        ref_emb0 = init_ref_emb0
        ref_emb1 = init_ref_emb1
        ref_pos0 = init_ref_pos0
        ref_pos1 = init_ref_pos1
        # online机制的队列
        ref_cache = {
            "mem": [],
            "reg": [],
            "pos": []
        }

        print('=====Step 2: Run sequence=====')
        for frame_id, imf in enumerate(tqdm(ims)):
            # Preprocess
            # print(f'Frame: #{frame_id} {imf}')

            image_ori = cv2.imread(os.path.join(img_dir, imf))
            image = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
            H, W = image.shape[:2]
            img_crop, resize_factor, att_mask = sample_target(image, state)

            img = torch.tensor(img_crop).permute((2, 0, 1)).unsqueeze(dim=0)
            # 归一化
            # mean = torch.tensor(mean).view((1, 3, 1, 1))
            # std = torch.tensor(std).view((1, 3, 1, 1))
            # img_tensor = torch.tensor(img_crop).float().permute((2, 0, 1)).unsqueeze(dim=0)
            # img = ((img_tensor / 255.0) - mean) / std  # (1,3,H,W)
            
            # Deal with the attention mask
            mask = torch.from_numpy(att_mask).to(torch.bool).unsqueeze(dim=0)  # (1,H,W)

            mask = F.interpolate(mask[None].float(), size=[20, 20]).to(torch.bool)[0]
            pos = positionEmbeddingSine(128, mask).flatten(2).permute(2, 0, 1)  # HWxBxC
            inr = positionEmbeddingSine(32, mask).flatten(2).permute(2, 0, 1)  # HWxBxC
            inr_emb = torch.repeat_interleave(inr, 4, dim=1).transpose(0, -1).reshape(64, -1, 400).transpose(0, -1)

            if stage in ["a","g"] and netinfo_net.ImageMake_on:
                img = img.permute((0,2,3,1)).contiguous().numpy().astype(np.uint8) # NHWC输入
                # img = img.contiguous().numpy() # ftmp-NCHW输入
                pos = pos.contiguous().numpy().astype(np.float32)
                inr_emb = inr_emb.contiguous().numpy().astype(np.float32)
            else:
                img = img.permute((0,2,3,1)).contiguous().numpy().astype(np.float32) # NHWC输入
                # img = img.contiguous().numpy().astype(np.float32) # ftmp-NCHW输入
                pos = pos.contiguous().numpy().astype(np.float32)
                inr_emb = inr_emb.contiguous().numpy().astype(np.float32)

            # 构造Icraft tensor
            inputs=[]
            # inputs.append(Tensor(img, Layout("NHWC")))#[1,320,320,3] # 图片输入，带imk
            inputs.append(Tensor(img, Layout("***C")))#[1,3,320,320] # ftmp输入
            inputs.append(Tensor(pos, Layout("**C")))#[400, 1, 256]
            inputs.append(Tensor(inr_emb, Layout("**C")))#[400, 4, 64]
            inputs.append(Tensor(ref_mem0, Layout("**C")))#[400, 1, 256]
            inputs.append(Tensor(ref_mem1, Layout("**C")))#[1600, 1, 256]
            inputs.append(Tensor(ref_emb0, Layout("**C")))#[400, 1, 256]
            inputs.append(Tensor(ref_emb1, Layout("**C")))#[1600, 1, 256]
            inputs.append(Tensor(ref_pos0, Layout("**C")))#[400, 4, 64]
            inputs.append(Tensor(ref_pos1, Layout("**C")))#[400, 16, 64]

            # dma init(if use imk)
            dmaInit(run_sim, netinfo_net.ImageMake_on, netinfo_net.i_shape[0][1:], inputs[0], device)

            # 前向
            generated_output = session.forward(inputs)

            if not run_sim:
                device.reset(1)
                if calctime:
                    calctime_detail(session, network, name="./"+network.name+"_time.xlsx")

            # 后处理
            search_mem, bbox_coor, iou_feat = generated_output
            if frame_id:
                bbox_coor = np.array(bbox_coor).reshape(-1)
                outputs_coord = box_xyxy_to_xywh(bbox_coor)
                out_dict = box_xyxy_to_cxcywh(bbox_coor)
                pred_boxes = out_dict.view(-1, 4)
                # Baseline: Take the mean of all predicted boxes as the final result
                pred_box = (pred_boxes.mean(dim=0) * SEARCH_SIZE / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
                # Get the final box result
                state = clip_box(map_box_back(state, pred_box, resize_factor), H, W, margin=10)
            else:
                bbox_coor = np.array([0., 0., 0., 0.], dtype=np.float32)
                outputs_coord = transform_image_to_crop(torch.Tensor(state), torch.Tensor(state), resize_factor)
                # np.savetxt(save_dir+"/search_mem.txt", np.array(search_mem)[:, 0, :])
            
            # update cache
            ref_list, ref_cache = update_ref(inr, torch.Tensor(np.array(search_mem)), outputs_coord, ref_cache)
            
            ref_mem0, ref_mem1, ref_reg0, ref_reg1, ref_pos0, ref_pos1 = ref_list
            ref_emb0 = torch.bmm(ref_reg0, embed_bank).transpose(0, 1).contiguous().numpy().astype(np.float32)
            ref_emb1 = torch.bmm(ref_reg1, embed_bank).transpose(0, 1).contiguous().numpy().astype(np.float32)

            # Record tracking results
            tracking_result.append(state)
            print(state)
            if show:
                # 绘制边界框
                cv2.rectangle(image_ori, (int(state[0]), int(state[1])), (int(state[0]+state[2]), int(state[1]+state[3])), (0, 255, 0), 2)
                # 显示图像
                cv2.imshow('Image with Bounding Box', image_ori)
                cv2.waitKey(1)
        if save:
            save_bb(resfn, tracking_result)
            print(f"Tracking results of the seq '{test_seq}' saved to {resfn}.")

    # 关闭设备
    Device.Close(device)
    
