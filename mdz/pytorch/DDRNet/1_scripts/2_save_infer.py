import sys
sys.path.append("./tools")
import argparse
import os
import logging
import timeit
import numpy as np
from PIL import Image
import torch
import _init_paths
import models
import datasets
import cv2
from config import config
from config import update_config
from utils.utils import create_logger, get_confusion_matrix
from torch.nn import functional as F
from yacs.config import CfgNode

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="experiments/cityscapes/ddrnet23.yaml",
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--img", default="../2_compile/qtset/cityscapes/frankfurt_000000_000294_leftImg8bit.png", type=str, help="image path")
    parser.add_argument('--model_pt', type=str, default=R"../2_compile/fmodel/DDRNet_1024x2048_traced.pt", help='torchscript model path')
    args = parser.parse_args()
    update_config(config, args)
    return args

def convert_label(config, label, inverse=False):

    ignore_label = config.TRAIN.IGNORE_LABEL
    label_mapping = {-1: ignore_label, 0: ignore_label, 
                              1: ignore_label, 2: ignore_label, 
                              3: ignore_label, 4: ignore_label, 
                              5: ignore_label, 6: ignore_label, 
                              7: 0, 8: 1, 9: ignore_label, 
                              10: ignore_label, 11: 2, 12: 3, 
                              13: 4, 14: ignore_label, 15: ignore_label, 
                              16: ignore_label, 17: 5, 18: ignore_label, 
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                              25: 12, 26: 13, 27: 14, 28: 15, 
                              29: ignore_label, 30: ignore_label, 
                              31: 16, 32: 17, 33: 18}
    temp = label.copy()
    if inverse:
        for v, k in label_mapping.items():
            label[temp == k] = v
    else:
        for k, v in label_mapping.items():
            label[temp == k] = v
    return label

def get_palette(n):
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def one_image_forward(args,config, test_dataset, model, sv_dir='', sv_pred=False):
    model.eval()
    confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    # config.MODEL.NUM_OUTPUTS = 1#将输出置为1
    image_path = args.img
    name = image_path.split("/")[-1].split(".")[0]
    image = cv2.imread(image_path, cv2.IMREAD_COLOR) #(1024, 2048, 3)  
    image = image.astype(np.float32)[:, :, ::-1]  #(1024, 2048, 3)
    image = image.transpose((2, 0, 1))  # (3, 1024, 2048)
    image = image / 255.0
    image[0] = (image[0] - 0.485) / 0.229
    image[1] = (image[1] - 0.456) / 0.224
    image[2] = (image[2] - 0.406) / 0.225

    image = torch.tensor(image)
    image = image.unsqueeze(0)  #(1, 3, 1024, 2048)

    label = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)   #(1024, 2048)
    label = convert_label(config, label) #(19, 1024, 2048)
    label = torch.tensor(label)
    label = label.unsqueeze(0) #(1, 19, 1024, 2048)
    size = label.size()
    
    pred = test_dataset.multi_scale_inference(
                config,
                model,
                image,
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST)  #(1, 19, 1024, 2048)#config.TEST.FLIP_TEST

    if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

    confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

    if sv_pred:
        palette = get_palette(256)
        pred = np.asarray(np.argmax(pred.detach().cpu(), axis=1), dtype=np.uint8) #(1, 19, 1024, 2048) -> (1, 1024, 2048)
        pred = convert_label(config, pred, inverse=True)
        pred = pred.squeeze(0) 
        save_img = Image.fromarray(pred)
        save_img.putpalette(palette)
        if not os.path.isdir(sv_dir):
            os.mkdir(sv_dir)
        save_path =os.path.join(sv_dir, name+'_save_infer.png')
        print("save results in:",save_path)
        save_img.save(save_path)

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc


def main():
    args = parse_args()
    logger, final_output_dir, _ = create_logger(config, args.cfg, 'test')
    new_config = CfgNode()
    new_config.MODEL = CfgNode()
    new_config.MODEL.NUM_OUTPUTS = 1
    # 合并新配置到原配置
    config.merge_from_other_cfg(new_config)
    # load model
    model = torch.jit.load(args.model_pt)

    # prepare data
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
                        num_samples=None,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size,
                        downsample_rate=1)

    start = timeit.default_timer()
    
    # inference
    mean_IoU, IoU_array, pixel_acc, mean_acc = one_image_forward(args,config, 
                                                                 test_dataset, 
                                                                 model,
                                                                 sv_dir="./output",
                                                                 sv_pred=True)

    msg = 'MeanIU: {: 4.4f}, Pixel_Acc: {: 4.4f}, \
        Mean_Acc: {: 4.4f}, Class IoU: '.format(mean_IoU, 
        pixel_acc, mean_acc)
    logging.info(msg)
    logging.info(IoU_array)

    end = timeit.default_timer()
    logger.info('Mins: %d' % int((end-start)/60))
    logger.info('Done')


if __name__ == '__main__':
    main()
