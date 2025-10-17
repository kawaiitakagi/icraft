import torch
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
import logging
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
        Yaml_Path = "../cfg/TSN_psin_test.yaml"
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
    #不支持adapt阶段
    if stage =="a":
        print("此脚本暂不支持adapt阶段,其余阶段可上板or仿真")
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
    video_list_txt = os.path.abspath(cfg["dataset"]["list"])
    resRoot = cfg["dataset"]["res"]
    label_txt = cfg["dataset"]["label_txt"]
    if not os.path.exists(resRoot):
        os.makedirs(resRoot)

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
    
    

    lines = read_list_from_txt(video_list_txt)
    data_list = load_data_list(lines)#[dict]

    results = []
    for data in tqdm(data_list):
        video_name = data["filename"]
        gt_label = data["label"]
        cur_video_path = video_path +"/"+video_name
        # print(cur_video_path)
        cap = cv2.VideoCapture(cur_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_per_segment = total_frames // num_segments
        
        net1_outs = []
        for index in range(num_segments):
            #===== tsn-net1前向 ======#
            start_frame = index * frames_per_segment
            end_frame = start_frame + frames_per_segment
            # cap.set(cv2.CAP_PROP_POS_FRAMES, np.random.randint(start_frame, end_frame-1))#随机抽取7帧
            cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame)

            ret, frame = cap.read()

            if ret:
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

                if stage=="g":
                    dest_base_addr = net2_input_chunk.begin.addr() + offset*index
                    dest_end_addr = net2_input_chunk.begin.addr() + offset*(index+1)
                    plddr_memcpy(src_base_addr, src_end_addr, dest_base_addr, dest_end_addr, device)
                else:
                    outputs = np.array(generated_output[0])#[1,2048]
                    net1_outs.append(torch.tensor(outputs))
                if not run_sim:
                    device.reset(1)
                    # calctime_detail(session_1, network1_view, name="./tsn_net1_time.xlsx")
        cap.release()  
        if stage=="p" or stage=="o" or stage=="q":
            net1_out = torch.cat(net1_outs) #[7,2048], 拼接net1输出  
            net2_input = net1_out.numpy().astype(np.float32).reshape(num_segments,2048)
            # 构造Icraft tensor
            input_tensor=Tensor(net2_input, Layout("*C"))

        #===== tsn-net2前向 ======#
        generated_output_net2 = session_2.forward([input_tensor])
        for tensor in generated_output_net2:
            timeout = timedelta(milliseconds=1000)
            tensor.waitForReady(timeout)
        
        if not run_sim:
            device.reset(1)
            # calctime_detail(session_2, network2_view, name="././tsn_net2_time.xlsx")
        
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
        result = {"pred":pred_scores,"label":gt_label}# 用于精度测试
        results.append(result)

        # 读取label文件，获取分类类别名称
        # labels = open(label_txt).readlines()
        # labels = [x.strip() for x in labels]
        # top1_label = labels[int(max_idx)]
        # top1_score = pred_scores[int(max_idx)]
        # print("[Top1] label:{}, score:{}".format(top1_label,top1_score))

        # 排序找到top5类别  
        # score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
        # score_sorted = sorted(score_tuples, key=lambda x: x[1], reverse=True)
        # top5_tuples = score_sorted[:5]
        # print('The top-5 labels with corresponding scores are:')
        # for idx,tuple_tmp in enumerate(top5_tuples):   
        #     print(str(labels[tuple_tmp[0]])+": ",tuple_tmp[1])

        # cur_img = img_ori
        # cv2.putText(cur_img, top1_label, (5,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        # if show:
        #     cv2.imshow("results", cur_img)
        #     cv2.waitKey(1)
        # if save:
        #     video_name =video_name.split(".")[0]
        #     save_path = resRoot + '/'+ video_name +"_result.png"
        #     cv2.imwrite(save_path, cur_img)

    # 精度测试
    eval_results = compute_metrics(results)
    print("acc/top1: {}, acc/top5: {}, acc/mean1: {}".format(eval_results["top1"], eval_results["top5"], eval_results["mean1"]))
    log_name = "TSN_metrics.log"
    logging.basicConfig(filename=log_name, level=logging.INFO, format='%(message)s')
    logging.info("Top1 acc: {}%".format(eval_results["top1"]*100))
    logging.info("Top5 acc: {}%".format(eval_results["top5"]*100))
    logging.info("Mean acc: {}%".format(eval_results["mean1"]*100))
    print("results save in ", log_name)
    # 关闭设备
    Device.Close(device)
    # 关闭设备
    Device.Close(device)
    
