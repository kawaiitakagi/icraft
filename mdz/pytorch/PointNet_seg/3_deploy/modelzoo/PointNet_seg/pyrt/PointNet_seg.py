import torch 
import numpy as np 
import matplotlib.pyplot as plt
import sys 
import yaml
from tqdm import tqdm
import logging
# 配置logging基本设置
logging.basicConfig(level=logging.INFO)
# Import PointNet_seg_utils
from PointNet_seg_utils import showpoints,ShapeNetDataset
# Import icraft packages
from icraft.xir import *
from icraft.xrt import *
from icraft.host_backend import *
from icraft.buyibackend import *
# Import pyrt dependencies
sys.path.append(R"../../../Deps/modelzoo")
from pyrtutils.Netinfo import Netinfo
from pyrtutils.utils import *
from pyrtutils.et_device import *
from pyrtutils.icraft_utils import *
from pyrtutils.calctime_utils import *
def main(config_path) -> None:
    
    # 从yaml里读入配置
    cfg = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
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
    timeAyalysis = cfg["imodel"]['timeRes']
    metricRes = cfg["imodel"]['metricRes']
    imgRoot = os.path.abspath(cfg["dataset"]["dir"])
    resRoot = cfg["dataset"]["res"]
    # check resRoot
    if os.path.exists(resRoot):
        print(f"The directory {resRoot} already exists.")
    else:
        os.makedirs(resRoot)
        print(f"The directory {resRoot} has been successfully created !")
    # param
    class_choice = cfg["param"]["class_choice"]
    # 加载network
    network = loadNetwork(JSON_PATH, RAW_PATH)
    print('INFO: Load network!')
    # 初始化netinfo
    netinfo = Netinfo(network)
    # 选择对网络进行切分
    network_view = network.view(0)
    # 打开device
    device = openDevice(run_sim, ip, netinfo.mmu or load_mmu)
    print('INFO: Open Device!')
    # 初始化session
    session = initSession(run_sim, network_view, device, netinfo.mmu or load_mmu, load_speedmode, load_compressFtmp)
    #开启计时功能
    session.enableTimeProfile(True)
	#session执行前必须进行apply部署操作
    session.apply()
    print('INFO: Session Apply!')
    
    # prepare test dataset
    d = ShapeNetDataset(
        root=imgRoot,
        class_choice=[class_choice],
        split='test',
        data_augmentation=False)
    dataset = ShapeNetDataset(
        root=imgRoot,
        classification=False,
        class_choice=[class_choice])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        drop_last=True,
        shuffle=True,
        num_workers=int(1))
    test_dataset = ShapeNetDataset(
        root=imgRoot,
        classification=False,
        class_choice=[class_choice],
        split='test',
        data_augmentation=False)
    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        drop_last=True,
        shuffle=True,
        num_workers=int(1))
    
    num_classes = dataset.num_seg_classes
    
    blue = lambda x: '\033[94m' + x + '\033[0m'
    logging.info(
        f"DATA INFO:\n"
        f"\tData Root Path:                {imgRoot}\n"
        "\n"
        f"Simulation Mode:\t                 {run_sim}\n"
        "\n"
        f"TEST INFO:\n"
        f"\tTested Model Path:            {JSON_PATH}\n"
        "\n"
        f"\tTrain:            {len(dataset)}\n"
        "\n"
        f"\tTest:             {len(test_dataset)}\n"
        "\n"
        f"\tTest_class:       {class_choice}\n"
        "\n"
        f"\tSeg_parts:        {num_classes}\n"
        "\n"
    )
    class_choice
    # calculate mIOU & show seg results
    shape_ious = []
    for idx, data in tqdm(enumerate(testdataloader, 0)):
        
        points, target = data
        target_np = target.cpu().data.numpy() - 1
        points = points[0]
        target = target[0]
        point_np = points.numpy()
        point = points.transpose(1, 0).unsqueeze(0).contiguous()
        # prepare input_data
        _data = point.numpy()#(1,3,2500)
        input_tensor = Tensor(_data, Layout("**C"))
        # 模型前向推理
        generated_output = session.forward([input_tensor])
        # 前向完，重置device
        if not run_sim: 
            device.reset(1)
        # check outputs
        # [1,2500,N_CLASS] : pred_results 
        # [1,3,3]  : transform_matrix_3x3
        # [1,64,64]: transform_matrix_64x64
        for i in range(3):
            out = np.array(generated_output[i])
        #     print(out.shape)
        # print('INFO: get forward results!')
        
        # 组装成检测结果
        outputs = []
        for i in range(3):
            out = np.array(generated_output[i])
            outputs.append(torch.tensor(out))
        pred = outputs[0]
        pred_choice = pred.data.max(2)[1]
        pred_np = pred_choice.cpu().data.numpy()
       
        
        # visualize results
        if(show):
            cmap = plt.cm.get_cmap("hsv", 10)
            cmap = np.array([cmap(i) for i in range(10)])[:, :3]
            
            gt = cmap[target.numpy() - 1, :]#真值结果
            pred_color = cmap[pred_choice.numpy()[0], :]# 预测结果
            
            showpoints(xyz=point_np, c_gt=gt,c_pred=pred_color)
            print('INFO: Visualize pred results!')

        
        # if save, save each seg results
        if(save):
            res_path = resRoot + "/" + str(idx) + '.txt'
            np.savetxt(res_path,pred_np[0],fmt="%d")
        

        # calculate iou
        for shape_idx in range(target_np.shape[0]):
            parts = range(num_classes)  # np.unique(target_np[shape_idx])
            
            part_ious = []
            for part in parts:
                I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                if U == 0:
                    iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
                else:
                    iou = I / float(U)
                part_ious.append(iou)
            shape_ious.append(np.mean(part_ious))
    sResult = "mIOU for class {}: {}".format(class_choice, np.mean(shape_ious))
    # if metricRes, save acc to metric.log
    if metricRes:
        
        # save acc to metric.log
        ACC_PATH = R"./PointNet_seg_metrics.log"
        with open (ACC_PATH,'w') as f:
            f.write(f"mIoU: {np.mean(shape_ious) * 100}%")
            f.close()
    print(sResult)
    # 模型时间信息统计
    if not run_sim and timeAyalysis:
        print("*"*40,"TIME RESULTS","*"*40)
        TIME_PATH = R'./time_results.xlsx'
        calctime_detail(session,network_view, name=TIME_PATH)#获取时间信息并保存时间结果
        print('Time result save at',TIME_PATH)
    # 关闭设备
    Device.Close(device) 

if __name__ == '__main__':
   
    
    # Yaml_Path = '../cfg/pointnet_seg.yaml'
    Yaml_Path = sys.argv[1]# config_path: ../cfg/pointnet_seg.toml
    if len(sys.argv) < 2:
        print("Info:未传入yaml参数,读入默认yaml文件:‌{}进行相关配置.".format(Yaml_Path), VERBOSE, 0)        
    if len(sys.argv) == 2:
        Yaml_Path = sys.argv[1]
        print("info:传入yaml文件:‌{}进行相关配置.".format(Yaml_Path), VERBOSE, 0)        
    if len(sys.argv) > 2:
        print("info:传入参数数量错误,请检查运行命令!", VERBOSE, 0)        
        sys.exit(1)
    main(Yaml_Path)