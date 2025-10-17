'''
Author: chen dong
Date: 2024-11-27 16:46:52
LastEditors: chen dong
LastEditTime: 2024-12-25 10:21:00
Description: 
FilePath: \1_scripts\0_infer.py
'''

import numpy as np
import os
import sys
import os.path as osp

sys.path.append(R"../0_strongReid")
import torchvision.transforms as T

import argparse
import torch
import torch.onnx
from PIL import Image
import PIL.Image as pil_image
from PIL import ImageDraw
from config import cfg
from modeling import build_model
from modeling.baseline import Baseline
normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


IMG_PATH = "../3_deploy/modelzoo/strongReid/io/input"

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


transform = T.Compose([
    T.Resize([256, 128]),
    T.ToTensor(),
    normalize_transform
])

# def new_forward(self, x):

#     global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)
#     global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

#     if self.neck == 'no':
#         feat = global_feat
#     elif self.neck == 'bnneck':
#         feat = self.bottleneck(global_feat)  # normalize for angular softmax

    
#     cls_score = self.classifier(feat)
#     return cls_score, global_feat  # global feature for triplet loss

#     if self.training:
#         cls_score = self.classifier(feat)
#         return cls_score, global_feat  # global feature for triplet loss
#     else:
#         if self.neck_feat == 'after':
#             # print("Test with feature after BN")
#             return feat
#         else:
#             # print("Test with feature before BN")
#             return global_feat
# Baseline.forward = new_forward
def main():

    from utils.data_infer import make_val_data_loader
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="iconfigs/softmax_triplet_with_center_self.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg['DATASETS']['ROOT_DIR'] = IMG_PATH
    cfg.freeze()

    prenum_classes = 751
    val_loader, num_query, num_classes = make_val_data_loader(cfg)
    model = build_model(cfg, prenum_classes)
    checkpoint = torch.load(cfg.TEST.WEIGHT, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    
    alldata = []
    allfeats = []
    allpids = []
    allcamids = []
    for batch in val_loader:
        data, pids, camids = batch
        im = transform(data[0]).unsqueeze(0)
        feat = model(im)
        alldata.append(data[0])
        allfeats.append(feat)
        allpids.extend(np.asarray(pids))
        allcamids.extend(np.asarray(camids))

    feats = torch.cat(allfeats, dim=0)
    if cfg.TEST.FEAT_NORM == 'yes':
        print("The test feature is normalized")
        feats = torch.nn.functional.normalize(feats, dim=1, p=2)
    # query
    qf = feats[:num_query]
    # gallery
    gf = feats[num_query:]
    m, n = qf.shape[0], gf.shape[0]
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    print("query2gallery_dist:",distmat.cpu().detach().numpy()[0])
    top3_idx = torch.topk(distmat, k=3, largest=False).indices.cpu().numpy()
    selected_images = []

    for idx in top3_idx[0]:
        selected_images.append(alldata[idx+ 1])

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
    canvas.show()
    canvas.save("output_0_infer.png")

if __name__ == '__main__':
    main()
