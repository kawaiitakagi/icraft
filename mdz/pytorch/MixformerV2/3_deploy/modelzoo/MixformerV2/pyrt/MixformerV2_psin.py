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
from process_mixformerv2 import *
from pyrtutils.icraft_utils import *
from pyrtutils.Netinfo import Netinfo
from pyrtutils.utils import *
from pyrtutils.et_device import *
from pyrtutils.calctime_utils import *


if __name__ == "__main__":
    # 获取yaml
    Yaml_Path = "../cfg/Mixformerv2.yaml"
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

    # 网络参数
    TEMPLATE_FACTOR = 2.0
    SEARCH_FACTOR = 4.5
    TEMPLATE_SIZE = 112
    SEARCH_SIZE = 224
    # 归一化系数
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean = torch.tensor(mean).view((1, 3, 1, 1))
    std = torch.tensor(std).view((1, 3, 1, 1))

    # 网络的常量输入
    g_pos_embed_t = np.fromfile('../../../../2_compile/qtset/mixformerv2-small-5seqs_per100//pos_embed_t_1_49_768.ftmp', dtype=np.float32).reshape(1,49,768)
    g_pos_embed_s = np.fromfile('../../../../2_compile/qtset/mixformerv2-small-5seqs_per100//pos_embed_s_1_196_768.ftmp', dtype=np.float32).reshape(1,196,768)
    g_reg_tokens  = np.fromfile('../../../../2_compile/qtset/mixformerv2-small-5seqs_per100//reg_tokens_1_4_768.ftmp', dtype=np.float32).reshape(1,4,768)
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
        print(f"{len(ims)} frames in total.")

        print('=====Step 1: Initialize state and reference cache=====')
        print(f'load initial state from {gt_file}')
        annos = np.loadtxt(gt_file, delimiter=',', dtype=np.float32)
        state = list(annos[0]) #初始框坐标

        print('=====Step 2: Run sequence=====')
        for frame_id, imf in enumerate(tqdm(ims)):
            # print(f'Frame: #{frame_id} {imf}')
            if frame_id == 0: #初始帧
                image_ori = cv2.imread(os.path.join(img_dir, imf))
                # 前处理
                image = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
                H, W = image.shape[:2]
                img_crop, resize_factor = sample_target(image, state, TEMPLATE_FACTOR, TEMPLATE_SIZE) # 裁图
                template = preprocess(img_crop, mean, std) # 归一化
                # online机制
                g_max_pred_score = -1.0
                g_template = template.contiguous().numpy()
                g_online_template = template.contiguous().numpy()
                g_online_max_template = template.contiguous().numpy()
                g_max_score_decay = 1.0
                tracking_result.append(state)
            else:
                image_ori = cv2.imread(os.path.join(img_dir, imf))
                # 前处理
                image = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
                H, W = image.shape[:2]
                img_crop, resize_factor = sample_target(image, state, SEARCH_FACTOR, SEARCH_SIZE) # 裁图
                search = preprocess(img_crop, mean, std) # 归一化


                # Prepare inputs
                if stage in ["a","g"] and netinfo_net.ImageMake_on:
                    # g_template = g_template.permute((0,2,3,1)).astype(np.uint8) NHWC输入
                    g_template = g_template.astype(np.float32) # ftmp-NCHW输入
                    g_online_template = g_online_template.astype(np.float32) # ftmp输入
                    search = search.contiguous().numpy().astype(np.float32) # ftmp输入
                else:
                    # g_template = g_template.permute((0,2,3,1)).astype(np.float32) # NHWC输入
                    g_template = g_template.astype(np.float32) # ftmp-NCHW输入
                    g_online_template = g_online_template.astype(np.float32)
                    search = search.contiguous().numpy().astype(np.float32)
                    
                # 构造Icraft tensor
                inputs=[]
                # inputs.append(Tensor(g_template, Layout("NHWC")))# NHWC输入
                inputs.append(Tensor(g_template, Layout("***C"))) # ftmp-NCHW输入
                inputs.append(Tensor(g_online_template, Layout("***C")))
                inputs.append(Tensor(search, Layout("***C")))
                if stage == "p":
                    # 输入常量折叠后可注释掉下面几行，但parse阶段始终会有6个输入
                    inputs.append(Tensor(g_pos_embed_s, Layout("**C")))
                    inputs.append(Tensor(g_pos_embed_t, Layout("**C")))
                    inputs.append(Tensor(g_reg_tokens, Layout("**C")))

                # dma init(if use imk)
                dmaInit(run_sim, netinfo_net.ImageMake_on, netinfo_net.i_shape[0][1:], inputs[0], device)

                # 前向
                generated_output = session.forward(inputs)

                if not run_sim:
                    device.reset(1)
                    if calctime:
                        calctime_detail(session,network, name="./"+network.name+"_time.xlsx")

                # 后处理
                pred_score,pred_box,state = postprocess(generated_output,resize_factor,image,state,SEARCH_SIZE)
                
                # 更新机制
                g_max_pred_score = g_max_pred_score * g_max_score_decay
                if pred_score > 0.5 and pred_score > g_max_pred_score:
                    print(f'On frame {frame_id}, pred_score={pred_score}, max_pred_score={g_max_pred_score}, state={state}, updated')
                    z_patch, _ = sample_target(image, state, TEMPLATE_FACTOR, TEMPLATE_SIZE)
                    # cv2.imwrite(f'./debug/{frame_id}_online_max_template_{pred_score:.4f}.jpg', cv2.cvtColor(z_patch,cv2.COLOR_RGB2BGR)) # remember to change the color space
                    g_online_max_template = preprocess(z_patch, mean, std)
                    g_online_max_template = g_online_max_template.contiguous().numpy().astype(np.float32)
                    g_max_pred_score = pred_score
                if frame_id % INTERVAL == 0:
                    print(f"frame {frame_id}: update, pred_score={pred_score}, max_pred_score={g_max_pred_score}, state={state}")
                    g_online_template = g_online_max_template# update online template
                    g_max_pred_score = -1 # reset score
                    g_online_max_template = g_template
                # Record tracking results
                tracking_result.append(state)
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
    
