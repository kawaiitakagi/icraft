
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
import PIL.Image as pil_image
from PIL import ImageDraw
import torch
import numpy as np
from strongReid_utils import *

if __name__ == "__main__":
    # 获取yaml
    Yaml_Path = "../cfg/strongReid_psin.yaml"
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
    JSON_PATH, RAW_PATH = getJrPath(folderPath,stage,run_sim)

    cudamode = cfg["imodel"]["cudamode"]
    load_mmu = cfg["imodel"]["mmu"]
    load_speedmode = cfg["imodel"]["speedmode"]
    load_compressFtmp = cfg["imodel"]["compressFtmp"]
    ip = str(cfg["imodel"]["ip"])
    show = cfg["imodel"]["show"]
    save = cfg["imodel"]["save"]
    val = cfg["imodel"]["val"]

    imgRoot = cfg["dataset"]["dir"]
    imgList = os.path.abspath(cfg["dataset"]["list"])
    resRoot = cfg["dataset"]["res"]
    if not os.path.exists(resRoot):
        os.makedirs(resRoot)
    else:
        pass

    # 加载network
    network = loadNetwork(JSON_PATH, RAW_PATH)
  	# 初始化netinfo
    netinfo = Netinfo(network)
    # 选择对网络进行切分
    network_view = network.view(netinfo.inp_shape_opid + 1)
    # 打开device
    device = openDevice(run_sim, ip, netinfo.mmu or load_mmu,cudamode)
    # 初始化session
    session = initSession(run_sim, network_view, device, netinfo.mmu or load_mmu, load_speedmode, load_compressFtmp)
	#开启计时功能
    session.enableTimeProfile(True)
	#session执行前必须进行apply部署操作
    session.apply()

    if val:
        from utils.data import make_val_data_loader
        from utils.defaults import _C as model_cfg
        from utils.reid_metric import eval_func
        model_cfg.merge_from_file("iconfigs/softmax_triplet_with_center_self.yml")
        model_cfg['DATASETS']['ROOT_DIR'] = imgRoot
        model_cfg.freeze()
        num_classes = 751
        val_loader, num_query, num_classes = make_val_data_loader(model_cfg)
        allfeats = []
        allpids = []
        allcamids = []
        idx = 0
        for batch in val_loader:
            data, pids, camids = batch
            input_tensor = numpy2Tensor(np.expand_dims( np.array(data[0]),0).copy(),network)
            dmaInit(run_sim,netinfo.ImageMake_on, netinfo.i_shape[0][1:],input_tensor, device)
            output_tensors = session.forward([input_tensor])
            # print(output_tensors[0])
            if not run_sim: 
                device.reset(1)
                # calctime_detail(session,network, name="./"+network.name+"_time.xlsx")
            feat= torch.from_numpy(np.array(output_tensors[0]))
            allfeats.append(feat)
            allpids.extend(np.asarray(pids))
            allcamids.extend(np.asarray(camids))
            idx = idx +1
            print(idx)


        feats = torch.cat(allfeats, dim=0)
        if model_cfg.TEST.FEAT_NORM == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:num_query]
        q_pids = np.asarray(allpids[:num_query])
        q_camids = np.asarray(allcamids[:num_query])
        # gallery
        gf = feats[num_query:]
        g_pids = np.asarray(allpids[num_query:])
        g_camids = np.asarray(allcamids[num_query:])
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                    torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        print("The following is precision information")
        print(f'mAP:{mAP:.3f}')
        print("Above is precision information")

        with open(network.name+"_metrics.log",'w+') as f:
            f.write(f'mAP:{mAP * 100}%')

    else:
        from utils.data_infer import make_val_data_loader
        from utils.defaults import _C as model_cfg
        from utils.reid_metric import eval_func
        model_cfg.merge_from_file("iconfigs/softmax_triplet_with_center_self.yml")
        model_cfg['DATASETS']['ROOT_DIR'] = imgRoot

        model_cfg.freeze()
        num_classes = 751
        val_loader, num_query, num_classes = make_val_data_loader(model_cfg)
        alldata = []
        allfeats = []
        allpids = []
        allcamids = []
        CAL_TIME = True
        for batch in val_loader:
            data, pids, camids = batch
            input_tensor = numpy2Tensor(np.expand_dims( np.array(data[0]),0).copy(),network)
            dmaInit(run_sim,netinfo.ImageMake_on, netinfo.i_shape[0][1:],input_tensor, device)
            output_tensors = session.forward([input_tensor])
            # print(output_tensors[0])
            if not run_sim: 
                device.reset(1)
                if CAL_TIME:
                    calctime_detail(session,network, name="./"+network.name+"_time.xlsx")
                    CAL_TIME = False
            feat= torch.from_numpy(np.array(output_tensors[0]))
            alldata.append(data[0])
            allfeats.append(feat)
            allpids.extend(np.asarray(pids))
            allcamids.extend(np.asarray(camids))

        feats = torch.cat(allfeats, dim=0)
        if model_cfg.TEST.FEAT_NORM == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:num_query]
        q_pids = np.asarray(allpids[:num_query])
        q_camids = np.asarray(allcamids[:num_query])
        # gallery
        gf = feats[num_query:]
        g_pids = np.asarray(allpids[num_query:])
        g_camids = np.asarray(allcamids[num_query:])
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                    torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        print("query2gallery_dist:",distmat.cpu().detach().numpy()[0])

        top3_idx = torch.topk(distmat, k=3, largest=False).indices.cpu().numpy()
        selected_images = []

        for idx in top3_idx[0]:
            selected_images.append(alldata[idx+ 1])

        #获取对应的图片

        # 计算画布大小
        image_width, image_height = selected_images[0].size  # 假设所有图片大小相同
        canvas_width = image_width * 3  # 3 张图片横向排列
        canvas_height = image_height * 2  # 上半部分留空，下半部分放图片

        # 创建空白画布
        canvas = pil_image.new("RGB", (canvas_width, canvas_height), color=(255, 255, 255))

        # 将图片粘贴到画布的下半部分
        for i, img in enumerate(selected_images):
            x = i * image_width  # 横向位置
            y = image_height  # 纵向位置（从下半部分开始）
            canvas.paste(img, (x, y))

        canvas.paste(alldata[0], (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text(xy=(0, 0), text='query', fill=(255, 0, 0))
        draw.text(xy=(0, image_height), text='top3_match', fill=(255, 0, 0))
        # 显示画布
        if show:
            canvas.show()
        if save:
        # 保存画布（可选）
            canvas.save(os.path.join(resRoot, "output.png") )

    if not run_sim: Device.Close(device)    
