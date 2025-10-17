import torch
import numpy as np
import argparse
from tqdm.autonotebook import tqdm
import os
import yaml
import sys
import logging
sys.path.append("../../../../0_HybridNets")
sys.path.append(R"../../../Deps/modelzoo")
from icraft.xir import *
from icraft.xrt import *
from icraft.host_backend import *
from icraft.buyibackend import *
from post_process_hybridnets import *
from datetime import timedelta
from pyrtutils.icraft_utils import *
from pyrtutils.Netinfo import Netinfo
from pyrtutils.utils import *
from pyrtutils.et_device import *
from pyrtutils.calctime_utils import *
from utils import smp_metrics
from utils.utils import ConfusionMatrix, postprocess, scale_coords, process_batch, ap_per_class, fitness, \
    save_checkpoint, DataLoaderX, BBoxTransform, ClipBoxes, boolean_string, Params
from hybridnets.dataset import BddDataset
from torchvision import transforms
import torch.nn.functional as F
from utils.constants import *

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("Yaml_Path", type=str)
    ap.add_argument('-p', '--project', type=str, default='bdd100k', help='Project file that contains parameters')
    ap.add_argument('-bb', '--backbone', type=str,
                   help='Use timm to create another backbone replacing efficientnet. '
                   'https://github.com/rwightman/pytorch-image-models')
    ap.add_argument('-c', '--compound_coef', type=int, default=3, help='Coefficients of efficientnet backbone')
    ap.add_argument('-w', '--weights', type=str, default='../weights/hybridnets.pth', help='/path/to/weights')
    ap.add_argument('-n', '--num_workers', type=int, default=1, help='Num_workers of dataloader')
    ap.add_argument('--batch_size', type=int, default=1, help='The number of images per batch among all devices')
    ap.add_argument('-v', '--verbose', type=boolean_string, default=True, help='Whether to print results per class when valing')
    ap.add_argument('--cal_map', type=boolean_string, default=True, help='Calculate mAP in validation')
    ap.add_argument('--plots', type=boolean_string, default=True, help='Whether to plot confusion matrix when valing')
    ap.add_argument('--num_gpus', type=int, default=0, help='Number of GPUs to be used (0 to use CPU)')
    ap.add_argument('--conf_thres', type=float, default=0.001, help='Confidence threshold in NMS')
    ap.add_argument('--iou_thres', type=float, default=0.6, help='IoU threshold in NMS')

    args = ap.parse_args()
    return args



@torch.no_grad()
def val(val_generator, params, opt, seg_mode, session,stage,run_sim,netinfo,device,MULTICLASS, **kwargs):

    stats, ap, ap_class = [], [], []
    iou_thresholds = torch.linspace(0.5, 0.95, 10)#.cuda()  # iou vector for mAP@0.5:0.95
    num_thresholds = iou_thresholds.numel()
    names = {i: v for i, v in enumerate(params.obj_list)} # car
    nc = len(names)
    ncs = 1 if seg_mode == "binary" else len(params.seg_list) + 1
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    s_seg = ' ' * (15 + 11 * 8)
    s = ('%-15s' + '%-11s' * 8) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95', 'mIoU', 'mAcc')
    for i in range(len(params.seg_list)):
            s_seg += '%-33s' % params.seg_list[i]
            s += ('%-11s' * 3) % ('mIoU', 'IoU', 'Acc')
    p, r, f1, mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    iou_ls = [[] for _ in range(ncs)]
    acc_ls = [[] for _ in range(ncs)]
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    val_loader = tqdm(val_generator, ascii=True)
    for iter, data in enumerate(val_loader):
        img = data['img'] #[1,384,640,3]
        annot = data['annot'] #[1,19,5]
        seg_annot = data['segmentation'] #[1,384,640]
        shapes = data['shapes'] #origin img size:[720,1280]

        # icraft 模型推理
        input_img = np.ascontiguousarray(img).reshape(1,netinfo.i_shape[0][1],netinfo.i_shape[0][2],3)
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
        for tensor in output_tensors:
            timeout = timedelta(milliseconds=100000)
            tensor.waitForReady(timeout)

        if not run_sim:
            device.reset(1)

        out0 = output_tensors[0] # regeression
        out1 = output_tensors[1] # classification
        out2 = output_tensors[2] # seg
        regression, classification, segmentation = np.array(out0), np.array(out1), np.array(out2)#[1,46035,4]、[1,46035,1]、[1,384,640,3]
        regression, classification, segmentation = torch.tensor(regression), torch.tensor(classification), torch.tensor(segmentation).permute(0,3,1,2)

        # 后处理
        anchors = torch.Tensor(np.fromfile(f'../io/anchors_384x640.ftmp',np.float32).reshape(1,46035,4))
        img = img.permute(0,3,1,2)
        if opt.cal_map:
            out = postprocess(img.detach(),
                              torch.stack([anchors[0]] * img.shape[0], 0).detach(), regression.detach(),
                              classification.detach(),
                              regressBoxes, clipBoxes,
                              opt.conf_thres, opt.iou_thres)  # 0.5, 0.3

            for i in range(annot.size(0)):
                seen += 1
                labels = annot[i]
                labels = labels[labels[:, 4] != -1]

                ou = out[i]
                nl = len(labels)

                pred = np.column_stack([ou['rois'], ou['scores']])
                pred = np.column_stack([pred, ou['class_ids']])
                pred = torch.from_numpy(pred)#.cuda()

                target_class = labels[:, 4].tolist() if nl else []  # target class

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, num_thresholds, dtype=torch.bool),
                                      torch.Tensor(), torch.Tensor(), target_class))
                    continue

                if nl:
                    pred[:, :4] = scale_coords(img[i][1:], pred[:, :4], shapes[i][0], shapes[i][1])
                    labels = scale_coords(img[i][1:], labels, shapes[i][0], shapes[i][1])
                    correct = process_batch(pred, labels, iou_thresholds)
                    if opt.plots:
                        confusion_matrix.process_batch(pred, labels)
                else:
                    correct = torch.zeros(pred.shape[0], num_thresholds, dtype=torch.bool)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), target_class))

            if MULTICLASS:
                segmentation = segmentation.log_softmax(dim=1).exp()
                _, segmentation = torch.max(segmentation, 1)  # (bs, C, H, W) -> (bs, H, W)
            else:
                segmentation = F.logsigmoid(segmentation).exp()

            tp_seg, fp_seg, fn_seg, tn_seg = smp_metrics.get_stats(segmentation, seg_annot, mode=seg_mode,
                                                                   threshold=0.5 if not MULTICLASS else None,
                                                                   num_classes=ncs if MULTICLASS else None)
            iou = smp_metrics.iou_score(tp_seg, fp_seg, fn_seg, tn_seg, reduction='none')
            # print(iou)
            acc = smp_metrics.balanced_accuracy(tp_seg, fp_seg, fn_seg, tn_seg, reduction='none')

            for i in range(ncs):
                iou_ls[i].append(iou.T[i].detach().cpu().numpy())
                acc_ls[i].append(acc.T[i].detach().cpu().numpy())


    if opt.cal_map:
        for i in range(ncs):
            iou_ls[i] = np.concatenate(iou_ls[i])
            acc_ls[i] = np.concatenate(acc_ls[i])
        # print(len(iou_ls[0]))
        iou_score = np.mean(iou_ls)
        # print(iou_score)
        acc_score = np.mean(acc_ls)

        miou_ls = []
        for i in range(len(params.seg_list)):
            if seg_mode == BINARY_MODE:
                # typically this runs once with i == 0
                miou_ls.append(np.mean(iou_ls[i]))
            else:
                miou_ls.append(np.mean( (iou_ls[0] + iou_ls[i+1]) / 2))

        for i in range(ncs):
            iou_ls[i] = np.mean(iou_ls[i])
            acc_ls[i] = np.mean(acc_ls[i])

        # Compute statistics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]
        # print(stats[3])

        # Count detected boxes per class
        # boxes_per_class = np.bincount(stats[2].astype(np.int64), minlength=1)

        ap50 = None
        save_dir = 'plots'
        os.makedirs(save_dir, exist_ok=True)

        # Compute metrics
        if len(stats) and stats[0].any():
            p, r, f1, ap, ap_class = ap_per_class(*stats, plot=opt.plots, save_dir=save_dir, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=1)  # number of targets per class
        else:
            nt = torch.zeros(1)

        # Print results
        print(s_seg)
        print(s)
        pf = ('%-15s' + '%-11i' * 2 + '%-11.3g' * 6) % ('all', seen, nt.sum(), mp, mr, map50, map, iou_score, acc_score)
        for i in range(len(params.seg_list)):
            tmp = i+1 if seg_mode != BINARY_MODE else i
            pf += ('%-11.3g' * 3) % (miou_ls[i], iou_ls[tmp], acc_ls[tmp])
            if i == 0:
                road_mIoU = miou_ls[i]
            elif i == 1:
                ll_IoU = iou_ls[tmp]
                ll_acc = acc_ls[tmp]
        print(pf)

        # Print results per class
        if opt.verbose and nc > 1 and len(stats):
            pf = '%-15s' + '%-11i' * 2 + '%-11.3g' * 4
            for i, c in enumerate(ap_class):
                print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

        # Plots
        if opt.plots:
            confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
            confusion_matrix.tp_fp()

        results = (mp, mr, map50, map, iou_score, acc_score)
        
        # 保存精度测试结果
        log_name = "HybridNets_metrics.log"
        logging.basicConfig(filename=log_name, level=logging.INFO, format='%(message)s', force=True)
        logging.info("TOD-recall: {:.3f}%".format(mr*100))
        logging.info("TOD-map50: {:.3f}%".format(map50*100))
        logging.info("DAS-mIoU: {:.3f}%".format(road_mIoU*100))
        logging.info("LLD-Iou: {:.3f}%".format(ll_IoU*100))
        logging.info("LLD-acc: {:.3f}%".format(ll_acc*100))
        logging.shutdown()
        print("results save in ", log_name)


    # 关闭设备
    Device.Close(device)


    return 0


def forward_totensor(self,pic):
    # return F.to_tensor(pic)
    return torch.tensor(pic)# icraft内部进行归一化、/255
transforms.ToTensor.__call__ = forward_totensor

if __name__ == "__main__":

    #============== Icraft精度测试相关配置 ==============#
    # 获取yaml
    Yaml_Path = "../cfg/HybridNets_test.yaml"
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

    # 加载数据集
    datasetRoot = os.path.abspath(cfg["dataset"]["dir"])
    resRoot = cfg["dataset"]["res"]

    # 模型自身相关参数配置
    MULTICLASS = cfg["param"]["multiclass"]

    # 加载network
    network = loadNetwork(JSON_PATH, RAW_PATH)
  	# 初始化netinfo
    netinfo = Netinfo(network)
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

    #============== 源码精度测试相关配置 ==============#
    args = parse_args()
    compound_coef = args.compound_coef
    project_name = args.project
    # params = Params(f'../../../../0_HybridNets/projects/{project_name}.yml')
    params = Params(f'../cfg/bdd100k.yml')
    obj_list = params.obj_list
    seg_mode = "multilabel" if not MULTICLASS else "multiclass" if len(params.seg_list) > 1 else "binary"   

    # 根据HybridNets_test.yaml中指定的测试集路径加载数据集
    params.dataset["dataroot"] = os.path.join(datasetRoot, "imgs")
    params.dataset["labelroot"] = os.path.join(datasetRoot, "det_annot")
    params.dataset["segroot"][0] = os.path.join(datasetRoot, "da_seg_annot")
    params.dataset["segroot"][1] = os.path.join(datasetRoot, "ll_seg_annot")

    valid_dataset = BddDataset(
        params=params,
        is_train=False,
        inputsize=params.model['image_size'],
        transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=params.mean, std=params.std
            # ) # icraft内部已进行归一化
        ]),
        seg_mode=seg_mode
    )

    val_generator = DataLoaderX(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=params.pin_memory,
        collate_fn=BddDataset.collate_fn
    )

    val(val_generator, params, args, seg_mode, session,stage,run_sim,netinfo,device,MULTICLASS)
