import sys
sys.path.append(R"../../../Deps/modelzoo")
sys.path.append("../../../../0_YOLOP")
import torch
import numpy as np
import cv2
import os
import yaml
from icraft.xir import *
from icraft.xrt import *
from icraft.host_backend import *
from icraft.buyibackend import *
from tqdm import tqdm
from datetime import timedelta
from tqdm import tqdm
from pathlib import Path
from post_process_yolop import *
from pyrtutils.icraft_utils import *
from pyrtutils.Netinfo import Netinfo
from pyrtutils.utils import *
from pyrtutils.et_device import *
from pyrtutils.calctime_utils import *
from dataset import *
from utils import DataLoaderX
from lib.core.evaluate import ConfusionMatrix,SegmentationMetric
from lib.core.general import non_max_suppression,scale_coords,xyxy2xywh,xywh2xyxy,box_iou,ap_per_class
from lib.core.function import AverageMeter



def validate(val_loader, session, device,number_of_class,conf_thresh,iou_thresh,save_path,save,show):
    # setting
    save_conf=False # save auto-label confidences
    verbose=False
    save_hybrid=False
    nc = 1
    iouv = torch.linspace(0.5,0.95,10) #iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    seen =  0 
    names = {0:"0"}
    confusion_matrix = ConfusionMatrix(nc=number_of_class) #detector confusion matrix
    da_metric = SegmentationMetric(2) #segment confusion matrix    
    ll_metric = SegmentationMetric(2) #segment confusion matrix
 
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t_inf, t_nms = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    da_acc_seg = AverageMeter()
    da_IoU_seg = AverageMeter()
    da_mIoU_seg = AverageMeter()
    ll_acc_seg = AverageMeter()
    ll_IoU_seg = AverageMeter()
    ll_mIoU_seg = AverageMeter()
    T_inf = AverageMeter()
    T_nms = AverageMeter()

    stats, ap, ap_class = [], [], []
    for batch_i, (img, target, paths, shapes) in tqdm(enumerate(val_loader), total=len(val_loader)):
        assign_target = []
        for tgt in target:
            assign_target.append(tgt)
        target = assign_target
        nb, _, height, width = img.shape    #batch size, channel, height, width

        with torch.no_grad():
            pad_w, pad_h = shapes[0][1][1]
            pad_w = int(pad_w)
            pad_h = int(pad_h)
        
            # Icraft 前向推理
            img_icraft = img.permute(0,2,3,1) # NCHW->NHWC
            input_img = np.ascontiguousarray(img_icraft).reshape(1,netinfo.i_shape[0][1],netinfo.i_shape[0][2],3)
            if stage in ["a","g"] and netinfo.ImageMake_on:
                input_img = input_img.astype(np.uint8)
            else:
                input_img = input_img.astype(np.float32)

            # 构造Icraft tensor
            inputs = []
            inputs.append(Tensor(input_img, Layout("NHWC")))

            # dma init(if use imk)
            dmaInit(run_sim, netinfo.ImageMake_on, netinfo.i_shape[0][1:], inputs[0], device)

            # net1前向
            output_tensors = session.forward(inputs)

            # 手动同步
            for tensor in output_tensors:
                timeout = timedelta(milliseconds=50000)
                tensor.waitForReady(timeout)

            if not run_sim:
                device.reset(1)
                # calctime_detail(session, network, name="./"+network.name+"_time.xlsx")
            
            # 后处理
            out0 = output_tensors[0] # det_out1
            out1 = output_tensors[1] # det_out2
            out2 = output_tensors[2] # det_out3
            out3 = output_tensors[3] # da_seg_out
            out4 = output_tensors[4] # ll_seg_out
        
            det_out1, det_out2, det_out3 = np.transpose(np.array(out0), (0, 3, 1, 2)), np.transpose(np.array(out1), (0, 3, 1, 2)), np.transpose(np.array(out2), (0, 3, 1, 2))#[[1,18,80,80],[1,18,40,40],[1,18,20,20]]
            da_seg_out, ll_seg_out = np.transpose(np.array(out3), (0, 3, 1, 2)), np.transpose(np.array(out4), (0, 3, 1, 2)) # [1,2,640,640]、[1,2,640,640]
            det_outs = [torch.tensor(det_out1),torch.tensor(det_out2),torch.tensor(det_out3)]
            da_seg_out, ll_seg_out = torch.tensor(da_seg_out), torch.tensor(ll_seg_out) # [1,2,640,640]、[1,2,640,640]
           
            # 导出模型去除掉的det后处理部分
            strides = get_Stride(netinfo)
            nl = len(anchors)
            anchor_grid = torch.tensor(anchors).reshape((nl, 1, 3, 1, 1, 2))# shape(nl,1,na,1,1,2)
            pred = get_det_results(det_outs,anchor_grid,strides,nc+5) #[1, 25200, 6])

            # driving area segment evaluation
            _,da_predict=torch.max(da_seg_out, 1)
            _,da_gt=torch.max(target[1], 1)
            da_predict = da_predict[:, pad_h:height-pad_h, pad_w:width-pad_w]
            da_gt = da_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]

            da_metric.reset()
            da_metric.addBatch(da_predict.cpu(), da_gt.cpu())
            da_acc = da_metric.pixelAccuracy()
            da_IoU = da_metric.IntersectionOverUnion()
            da_mIoU = da_metric.meanIntersectionOverUnion()

            da_acc_seg.update(da_acc,img.size(0))
            da_IoU_seg.update(da_IoU,img.size(0))
            da_mIoU_seg.update(da_mIoU,img.size(0))

            # lane line segment evaluation
            _,ll_predict=torch.max(ll_seg_out, 1)
            _,ll_gt=torch.max(target[2], 1)
            ll_predict = ll_predict[:, pad_h:height-pad_h, pad_w:width-pad_w]
            ll_gt = ll_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]

            ll_metric.reset()
            ll_metric.addBatch(ll_predict.cpu(), ll_gt.cpu())
            ll_acc = ll_metric.lineAccuracy()
            ll_IoU = ll_metric.IntersectionOverUnion()
            ll_mIoU = ll_metric.meanIntersectionOverUnion()

            ll_acc_seg.update(ll_acc,img.size(0))
            ll_IoU_seg.update(ll_IoU,img.size(0))
            ll_mIoU_seg.update(ll_mIoU,img.size(0))

            # NMS
            target[0][:, 2:] *= torch.Tensor([width, height, width, height]) # to pixels
            lb = [target[0][target[0][:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            output = non_max_suppression(pred, conf_thres=conf_thresh, iou_thres=iou_thresh, labels=lb)

            if batch_i > 0:
                T_nms.update(t_nms/img.size(0),img.size(0))


        # Statistics per image
        for si, pred in enumerate(output):
            labels = target[0][target[0][:, 0] == si, 1:]     # all object in one image 
            nl = len(labels)    # num of object
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Append to text file
            if save:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(save_path+'/labels'+str(path.stem) + '.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')


            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels

                # Per target class
                for cls in torch.unique(tcls_tensor):                    
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        # n*m  n:pred  m:label
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices
                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    # Compute statistics
    # stats : [[all_img_correct]...[all_img_tcls]]
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy  zip(*) :unzip

    map70 = None
    map75 = None
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, save_dir=save_path, names=names)
        ap50, ap70, ap75,ap = ap[:, 0], ap[:,4], ap[:,5],ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map70, map75, map = p.mean(), r.mean(), ap50.mean(), ap70.mean(),ap75.mean(),ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12.3g' * 6  # print format
    # Print results per class
    if (verbose or (nc <= 20)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]

    detect_result = np.asarray([mp, mr, map50, map])*100
    da_segment_result = np.asarray([da_acc_seg.avg,da_IoU_seg.avg,da_mIoU_seg.avg])*100
    ll_segment_result = np.asarray([ll_acc_seg.avg,ll_IoU_seg.avg,ll_mIoU_seg.avg])*100


    return da_segment_result, ll_segment_result, detect_result
        




if __name__ == "__main__":
    # 获取yaml
    Yaml_Path = "../cfg/yolop_test.yaml"
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

    datasetRoot = os.path.abspath(cfg["dataset"]["dir"])
    cfg["dataset"]["data_root"] = os.path.join(datasetRoot, "imgs")
    cfg["dataset"]["labelroot"] = os.path.join(datasetRoot, "det_annot")
    cfg["dataset"]["da_segroot"] = os.path.join(datasetRoot, "da_seg_annot")
    cfg["dataset"]["ll_segroot"] = os.path.join(datasetRoot, "ll_seg_annot")
    resRoot = cfg["dataset"]["res"]
    save_path = resRoot
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 模型自身相关参数配置
    conf = cfg["param"]["conf"]
    iou_thresh = cfg["param"]["iou_thresh"]
    multilabel =  bool(cfg["param"]["multilabel"])
    number_of_class = int(cfg["param"]["number_of_class"])
    anchors = cfg["param"]["anchors"]
    # fpga_nms = bool(cfg["param"]["fpga_nms"])
    # seg_list = ["road","lane"]
    obj_list = ["car"]
    color_list_seg = {}
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(1)]

    # 加载network
    network = loadNetwork(JSON_PATH, RAW_PATH)
  	# 初始化netinfo
    netinfo = Netinfo(network)
    resized_shape = netinfo.i_shape[0][1:3]
    # 选择对网络进行切分
    network_view = network.view(0)
    # 打开device
    device = openDevice(run_sim, ip, netinfo.mmu or load_mmu)
    # 初始化session
    session = initSession(run_sim, network_view, device, netinfo.mmu or load_mmu, load_speedmode, load_compressFtmp)
	# 开启计时功能
    session.enableTimeProfile(True)
	# session执行前必须进行apply部署操作
    session.apply()

    # 加载数据集
    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # )
    valid_dataset = eval("BddDataset")(
        cfg=cfg,
        is_train=False,
        inputsize=resized_shape,
        # transform=transforms.Compose([
        #     # transforms.ToTensor(),
        # #     # normalize,# Icraft内部已进行归一化
        # ])
    )

    valid_loader = DataLoaderX(
        valid_dataset,
        batch_size=1,  #cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=0,  #cfg.WORKERS
        pin_memory=False,
        collate_fn=AutoDriveDataset.collate_fn
    )

    da_segment_results,ll_segment_results,detect_results = validate(valid_loader, session, device,number_of_class,conf,iou_thresh,save_path,save,show)

    # 打印精度测试结果
    print('Traffic Object Detection Result: MP:{:.4f}, MR:{:.4f}, Map50:{:.4f}, Map:{:.4f}'.format(detect_results[0], detect_results[1], detect_results[2], detect_results[3]))
    print('Drivable Area Segmentation Result: Acc:{:.4f}, Iou:{:.4f}, mIou:{:.4f}'.format(da_segment_results[0], da_segment_results[1], da_segment_results[2]))
    print('Lane Detection Result: Acc:{:.4f}, Iou:{:.4f}, mIou:{:.4f}'.format(ll_segment_results[0], ll_segment_results[1], ll_segment_results[2]))

    # 保存精度测试结果
    log_name = "YOLOP_metrics.log"
    logging.basicConfig(filename=log_name, level=logging.INFO, format='%(message)s', force=True)
    logging.info("TOD-recall: {:.4f}%".format(detect_results[1]))
    logging.info("TOD-map50: {:.4f}%".format(detect_results[2]))
    logging.info("DAS-mIoU: {:.4f}%".format(da_segment_results[2]))
    logging.info("LLD-Iou: {:.4f}%".format(ll_segment_results[1]))
    logging.info("LLD-acc: {:.4f}%".format(ll_segment_results[0]))
    logging.shutdown()
    print("results save in ", log_name)
       
    # 关闭设备
    Device.Close(device)
    
