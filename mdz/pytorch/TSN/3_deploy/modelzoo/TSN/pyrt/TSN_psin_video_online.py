import sys
sys.path.append(R"../../../Deps/modelzoo")
from icraft.xir import *
from icraft.xrt import *
from icraft.host_backend import *
from icraft.buyibackend import *
import numpy as np
import os
import yaml
import cv2
import re
from tqdm import tqdm
from datetime import timedelta
from process_tsn import *
from pyrtutils.icraft_utils import *
from pyrtutils.Netinfo import Netinfo
from pyrtutils.utils import *
from pyrtutils.et_device import *
from pyrtutils.calctime_utils import *

if __name__ == "__main__":
    # 获取yaml
    Yaml_Path = sys.argv[1]
    if len(sys.argv) < 2:
        Yaml_Path = "../cfg/TSN_psin_video.yaml"
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
    if stage !="g":
        print("此脚本仅支持BY阶段上板运行")
        sys.exit(1)
    run_sim = cfg["imodel"]["sim"]
    JSON_PATH, RAW_PATH= getJrPath(folderPath,stage,run_sim)
    
    load_mmu = cfg["imodel"]["mmu"]
    load_speedmode = cfg["imodel"]["speedmode"]
    load_compressFtmp = cfg["imodel"]["compressFtmp"]
    ip = str(cfg["imodel"]["ip"])
    save = cfg["imodel"]["save"]
    show = cfg["imodel"]["show"]

    video_path = os.path.abspath(cfg["dataset"]["dir"])
    # imgList = os.path.abspath(cfg["dataset"]["list"])
    resRoot = cfg["dataset"]["res"]
    label_txt = cfg["dataset"]["label_txt"]
    if not os.path.exists(resRoot):
        os.makedirs(resRoot)
    
    interval = cfg["param"]["interval"] #间隔抽取帧跑TSN
    num_segments = cfg["param"]["num_segments"]

    # 加载network
    network = loadNetwork(JSON_PATH, RAW_PATH)
  	# 初始化netinfo
    netinfo = Netinfo(network)

    # 选择对网络进行切分
    if stage=="p":
        network1_view = network.viewByOpId(0,126) #1,2048
        network2_view = network.viewByOpId(128,132) #7,2048
    elif stage=="o" or stage=="q":
        network1_view = network.viewByOpId(0,135) #1,2048
        network2_view = network.viewByOpId(136,132) #7,2048
    elif stage=="g":
        network1_view = network.viewByOpId(272,484) #1,64,32
        network2_view = network.viewByOpId(488,132) #1,7,64,32

    # 打开device
    device = openDevice(run_sim, ip, netinfo.mmu or load_mmu)
    # 初始化session
    session_1 = initSession(run_sim, network1_view, device, netinfo.mmu or load_mmu, load_speedmode, load_compressFtmp)
    session_2 = initSession(run_sim, network2_view, device, netinfo.mmu or load_mmu, load_speedmode, load_compressFtmp)
    buyibackend_1 = BuyiBackend(session_1.backends[0])
    buyibackend_2 = BuyiBackend(session_2.backends[0])
	#开启计时功能
    session_1.enableTimeProfile(True)
    session_2.enableTimeProfile(True)

	#session执行前必须进行apply部署操作
    session_1.apply()
    session_2.apply()

    if stage=="g":
        #计算net1输出和net2输入在PLDDR上的物理地址，用于搬数
        net1_output_segment = buyibackend_1.phy_segment_map[Segment(OUTPUT)]
        offset = net1_output_segment.byte_size
        src_base_addr = net1_output_segment.memchunk.begin.addr()
        src_end_addr = net1_output_segment.memchunk.begin.addr() + offset
        net2_input_chunk = buyibackend_2.phy_segment_map[Segment(INPUT)].memchunk

        bits = netinfo.ImageMake_.outputs[0].dtype.getStorageType().bits()
        # net2 fake input
        if bits == 8:
            input_shape =[1,num_segments,64,32]
            tensor_layout = icraft.xir.Layout("**Cc32")
            input_type = icraft.xir.TensorType(icraft.xir.IntegerType.SInt8(),input_shape,tensor_layout)
        else:
            input_shape =[1,num_segments,128,16]
            tensor_layout = icraft.xir.Layout("**Cc16")
            input_type = icraft.xir.TensorType(icraft.xir.IntegerType.SInt16(),input_shape,tensor_layout)
        input_tensor = icraft.xrt.Tensor(input_type,net2_input_chunk)
    
    cap = cv2.VideoCapture(video_path)
    window_length = num_segments  #窗口尺寸
    frame_count = 0  #视频总帧数
    forward_index = 0 # 参与前向的视频总帧数
    while cap.isOpened():
        ret, frame = cap.read()
        frame_count += 1
        if not ret:
            break

        # 间隔 interval 帧处理一张图片
        if frame_count % interval == 0:
            #===== tsn-net1前向 ======#
            index = forward_index % window_length + 1
            img_ori = frame
            img_rgb = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
            img_sz = img_ori.shape
            net1_sz = netinfo.i_shape[0][1:3]
            img = cv2.resize(img_rgb, net1_sz)

            if stage in ["a","g"] and netinfo.ImageMake_on:
                image = img.astype(np.uint8).reshape(netinfo.i_shape[0])
            else:
                image = img.astype(np.float32).reshape(netinfo.i_shape[0])

            # 构造Icraft tensor
            inputs=[]
            inputs.append(Tensor(image, Layout("NHWC")))
            
            # dma init(if use imk)
            dmaInit(run_sim, netinfo.ImageMake_on, netinfo.i_shape[0][1:], inputs[0], device)

            # net1前向
            generated_output = session_1.forward(inputs)
            for tensor in generated_output:
                timeout = timedelta(milliseconds=1000)
                tensor.waitForReady(timeout)

            dest_base_addr = net2_input_chunk.begin.addr() + offset*(index-1)
            dest_end_addr = net2_input_chunk.begin.addr() + offset*index
            plddr_memcpy(src_base_addr, src_end_addr, dest_base_addr, dest_end_addr, device)

            if not run_sim:
                device.reset(1)
                calctime_detail(session_1, network1_view, name="./tsn_net1_online_time.xlsx")
            if forward_index >= window_length-1:
                #===== tsn-net2前向 ======#
                generated_output_net2 = session_2.forward([input_tensor])
                for tensor in generated_output_net2:
                    timeout = timedelta(milliseconds=1000)
                    tensor.waitForReady(timeout)
        
                if not run_sim:
                    device.reset(1)
                    calctime_detail(session_2, network2_view, name="./tsn_net2_online_time.xlsx")
                
                # net2后处理
                net2_outputs = np.array(generated_output_net2[0])#(400,)
                cls_scores = net2_outputs[0].reshape(1,1,net2_outputs.size)#[1,1,400]
                # import torch.nn.functional as F
                # cls_scores = F.softmax(cls_scores, dim=2).mean(dim=1)
                # 等效替换为：先在第2维进行softmax操作
                softmax_scores = np.exp(cls_scores) / np.sum(np.exp(cls_scores), axis=2, keepdims=True)
                # 再计算在第1维的均值
                mean_scores = np.mean(softmax_scores, axis=1)
                max_idx = np.argmax(mean_scores,axis=-1)
                pred_scores = mean_scores.squeeze(0)
                # 读取label文件，获取分类类别名称
                labels = open(label_txt).readlines()
                labels = [x.strip() for x in labels]
                top1_label = labels[int(max_idx)]
                top1_score = pred_scores[int(max_idx)]
                print("[Frame:{}] [Top1] label:{}, score:{}".format(frame_count,top1_label,top1_score))

                # 排序找到top5类别  
                # score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
                # score_sorted = sorted(score_tuples, key=lambda x: x[1], reverse=True)
                # top5_tuples = score_sorted[:5]
                # print('The top-5 labels with corresponding scores are:')
                # for idx,tuple_tmp in enumerate(top5_tuples):   
                #     print(str(labels[tuple_tmp[0]])+": ",tuple_tmp[1])

                cur_img = img_ori
                cv2.putText(cur_img, top1_label, (5,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                video_name = video_path.split("\\")[-1].split(".")[0]

                if show:
                    cv2.imshow("results", cur_img)
                    cv2.waitKey(1)
                if save:
                    video_out_path =  resRoot + video_name
                    if not os.path.exists(video_out_path):
                            os.makedirs(video_out_path)
                    save_path = video_out_path+ '/frame_' + str(frame_count) + "_result.png"
                    cv2.imwrite(save_path, cur_img)
        forward_index += 1

    # 关闭设备
    Device.Close(device)
    
