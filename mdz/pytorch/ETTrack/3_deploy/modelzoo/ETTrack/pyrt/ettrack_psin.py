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
from tqdm import tqdm
from datetime import timedelta
from post_process_ettrack import *
from pyrtutils.icraft_utils import *
from pyrtutils.Netinfo import Netinfo
from pyrtutils.utils import *
from pyrtutils.et_device import *
from pyrtutils.calctime_utils import *

if __name__ == "__main__":
    # 获取yaml
    Yaml_Path = "../cfg/ettrack.yaml"
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
    folderPath_net1 = cfg["imodel"]["net1_dir"]
    folderPath_net2 = cfg["imodel"]["net2_dir"]
    stage = cfg["imodel"]["stage"]
    run_sim = cfg["imodel"]["sim"]
    JSON_PATH_net1, RAW_PATH_net1 = getJrPath(folderPath_net1,stage,run_sim)
    JSON_PATH_net2, RAW_PATH_net2 = getJrPath(folderPath_net2,stage,run_sim)

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

    #模型自身相关参数配置
    instance_size = cfg["param"]["instance_size"]

    # 跟踪器相关超参数
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    stride = 16
    image_sample_size = 256
    image_template_size = 128
    example_size = 127

    mean = np.array(mean)
    std = np.array(std)

    # 保存txt的路径
    save_dir = resRoot + "res_txt/"
    # 保存img的路径
    save_img_dir = resRoot + "res_img/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)


    # 加载network
    network_1 = loadNetwork(JSON_PATH_net1, RAW_PATH_net1)
    network_2 = loadNetwork(JSON_PATH_net2, RAW_PATH_net2)
  	# 初始化netinfo
    netinfo_net1 = Netinfo(network_1)
    netinfo_net2 = Netinfo(network_2)
    # 选择对网络进行切分
    network1_view = network_1.view(netinfo_net1.inp_shape_opid + 1)
    network2_view = network_2.view(netinfo_net2.inp_shape_opid + 1)
    # 打开device
    device = openDevice(run_sim, ip, netinfo_net1.mmu or netinfo_net2.mmu or load_mmu)
    # 初始化session
    session_1 = initSession(run_sim, network1_view, device, netinfo_net1.mmu or load_mmu, load_speedmode, load_compressFtmp)
    session_2 = initSession(run_sim, network2_view, device, netinfo_net2.mmu or load_mmu, load_speedmode, load_compressFtmp)

	#开启计时功能
    session_1.enableTimeProfile(True)
    session_2.enableTimeProfile(True)
	#session执行前必须进行apply部署操作
    session_2.apply()
    session_1.apply()


    # 加载数据集
    TEST_SET = read_list_from_txt(imgList)

    #================逐序列测试================#
    for test_seq in TEST_SET[:]:
        print("test_seq = ",test_seq)
        # ================准备测试集================#
        seq_cls = test_seq.split('-')[0]
        # 原LASOT数据集的测试集
        # seq_dir = imgRoot +f'/{seq_cls}/{test_seq}/'
        # gt_file = imgRoot +f'/{seq_cls}/{test_seq}/groundtruth.txt'
        # 内部LASOT_sub62试用以下格式：
        seq_dir = imgRoot +f'/{test_seq}/'
        gt_file = imgRoot +f'/{test_seq}/groundtruth.txt'
        img_dir = seq_dir + 'img/'
        print(f'loading groundtruth from {gt_file}')
        init_bbox = np.loadtxt(gt_file, delimiter=',',dtype=np.float32)
        tracking_result = []
        gt = init_bbox[0]
        tracking_result.append(gt)
        #================初始化算法参数================#
        avg_chans = None
        # 全局变量计算
        x,y,w,h = gt[0], gt[1], gt[2], gt[3]
        print(f"#{1} init_rect: x,y,w,h =[{x},{y},{w},{h}]")
        # 左上角转中心点
        cx = x+w/2
        cy = y+h/2
        target_pos =  np.array([cx,cy]) # 全局
        target_sz =  np.array([w,h]) # 全局

        print(f'loading image from {img_dir}')
        file_list = os.listdir(img_dir)
        file_list = sorted(file_list, key=numeric_sort_key)
        img_path = img_dir + file_list[0] # Initial frame
        print("load initial frame from ", img_path)
        img_ori = cv2.imread(img_path)
        img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        img_sz = img.shape
        H,W,_ = img.shape
        img_sample_sz = np.array([image_sample_size, image_sample_size])
        img_support_sz = img_sample_sz
        total_stride = stride

        score_sz = int(round(instance_size / total_stride))

        # 计算grids
        grid_to_search_x, grid_to_search_y = grids(score_sz, total_stride, instance_size)
        # create_hanning_window due to WINDOWING=='cosine'
        hanning_window = np.outer(np.hanning(score_sz), np.hanning(score_sz))
        #===== 开始追踪 ======#
        for id in tqdm(range(len(file_list))):
            # print("id ==== ",id)
            # if(not file_list[id].endswith('.png')):
            #     continue
            if id==0:
                # ======= net1前处理 ===========
                # crop_target
                target_img_crop, avg_chans = net1_preprocess(img, target_pos, target_sz)
                net1_input_sz = target_img_crop.shape
                # norm
                # ftmp
                # target_img_crop = np.transpose(target_img_crop, (2, 0, 1))#转换为CHW,ftmp输入才需要进行转换
                # target_img_norm = (target_img_crop / 255.0 - mean.reshape((3,1,1))) / std.reshape((3,1,1))
                # z = np.ascontiguousarray(target_img_norm).astype(np.float32).reshape(1,3,example_size, example_size)#[1,3,127,127]
                # NHWC
                z = np.ascontiguousarray(target_img_crop).reshape(1,example_size, example_size,3)#[1,127,127,3]
                if stage in ["a","g"] and netinfo_net1.ImageMake_on:
                    z = z.astype(np.uint8)
                else:
                    z = z.astype(np.float32)

                # 构造Icraft tensor
                net1_inputs = []
                net1_inputs.append(Tensor(z, Layout("NHWC")))

                # imk init
                if(netinfo_net1.ImageMake_on):
                    session_1.backends[0].initOp(netinfo_net1.ImageMake_)

                # dma init(if use imk)
                dmaInit(run_sim, netinfo_net1.ImageMake_on, netinfo_net1.i_shape[0][1:], net1_inputs[0], device)

                # net1前向
                net1_output_tensors = session_1.forward(net1_inputs)
                # for tensor in net1_output_tensors:
                #     timeout = timedelta(milliseconds=1000)
                #     tensor.waitForReady(timeout)

                res_out =  np.array(net1_output_tensors[0]).astype(np.float32)
                template =  np.ascontiguousarray(np.transpose(res_out, (0, 3, 1, 2)))

                if not run_sim:
                    device.reset(1)
                    if calctime:
                        calctime_detail(session_1,network_1, name="./"+network_1.name+"_time.xlsx")#计时
            else:
                # ======= net2前处理 ===========
                img_path = img_dir + file_list[id]
                print("load image from ",img_path)
                img_ori = cv2.imread(img_path)
                img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
                img_sz = img.shape
                # crop
                search_img_crop, scale_z = net2_preprocess(img,target_pos,target_sz, instance_size, avg_chans)
                # norm
                # ftmp
                # search_img_crop = np.transpose(search_img_crop, (2, 0, 1))#转换为CHW,ftmp输入才需要进行转换
                # search_img_norm=(search_img_crop / 255.0 - mean.reshape((3,1,1))) / std.reshape((3,1,1))
                # x = np.ascontiguousarray(search_img_norm).astype(np.float32).reshape(1,3,example_size, example_size)#[1,3,127,127]
                # NHWC
                x = np.ascontiguousarray(search_img_crop).reshape(1, instance_size, instance_size, 3)
                if stage in ["a","g"] and netinfo_net2.ImageMake_on:
                    x = x.astype(np.uint8)
                else:
                    x = x.astype(np.float32)

                # 构造Icraft tensor
                net2_inputs = []
                net2_inputs.append(Tensor(x, Layout("NHWC")))
                net2_inputs.append(Tensor(template, Layout("***C")))

                # imk init
                if(netinfo_net2.ImageMake_on and id==1):
                    session_2.backends[0].initOp(netinfo_net2.ImageMake_)

                # dma init(if use imk)
                dmaInit(run_sim,netinfo_net2.ImageMake_on, netinfo_net2.i_shape[0][1:],net2_inputs[0], device)
                
                # net2 前向
                net2_output_tensors = session_2.forward(net2_inputs)
               
                if not run_sim:
                    device.reset(1)
                    if calctime:
                        calctime_detail(session_2,network_2, name="./"+network_2.name+"_time.xlsx")#计时

                net2_out0 = net2_output_tensors[0]
                net2_out1 = net2_output_tensors[1]
                # 后处理
                cls_score, bbox_pred = np.array(net2_out0),np.array(net2_out1)
                cls_score = np.transpose(cls_score, (0, 3, 1, 2)) #[1,1,18,18]
                bbox_pred = np.transpose(bbox_pred, (0, 3, 1, 2))#[1,4,18,18]
                cls_score = sigmoid(cls_score).squeeze()

                # bbox to real predict
                bbox_pred = bbox_pred.squeeze()
                target_sz = target_sz*scale_z
                target_pos_tmp, target_sz_tmp, cls_score = postprocess(cls_score,bbox_pred,scale_z,target_pos,target_sz, grid_to_search_x, grid_to_search_y,hanning_window,instance_size)

                # 中心点坐标
                res_cx = max(0, min(img_sz[1], target_pos_tmp[0]))#w
                res_cy = max(0, min(img_sz[0], target_pos_tmp[1]))#h
                res_w = max(10, min(img_sz[1], target_sz_tmp[0]))
                res_h = max(10, min(img_sz[0], target_sz_tmp[1]))
                target_pos = np.array([res_cx, res_cy])
                target_sz = np.array([res_w, res_h])
                # 转换为左上角
                location = cxy_wh_2_rect(target_pos, target_sz)
                tracking_result.append(location)
                # print(f"target_pos={target_pos}, target_sz={target_sz}")
                print(f"#{id+1}, x,y,w,h = {location}")
                if save:
                    result_file = os.path.join(save_dir, test_seq +'.txt')
                    tracked_bb = np.array(tracking_result).astype(int)
                    np.savetxt(result_file, tracked_bb, delimiter='\t', fmt='%d')
                if show:
                     # 绘制边界框
                    cv2.rectangle(img_ori, (int(location[0]), int(location[1])), (int(location[0]+res_w), int(location[1]+res_h)), (0, 255, 0), 2)
                    
                    # 显示图像
                    cv2.imshow('Image with Bounding Box', img_ori)
                    cv2.waitKey(1)
                    # cv2.destroyAllWindows()

                device.reset(1)
    # 关闭设备
    Device.Close(device)
