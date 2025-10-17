import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import fbeta_score


def calculate_ssim(image1, image2):
    # 读取图像并转换为numpy数组
    image1 = np.array(image1)
    image2 = np.array(image2)
    # print(image1.shape, image2.shape)

    # 确保图像数据类型是float
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)

    # 计算SSIM
    ssim_value, _ = ssim(image1, image2, multichannel=True, full=True, win_size=3, data_range=255)
    return ssim_value


def calculate_mae(image1, image2):
    # 读取图像并转换为numpy数组
    image1 = np.array(image1)
    image2 = np.array(image2)

    # 计算像素差
    pixel_diff = np.abs(image1 - image2)

    # 计算平均绝对误差
    mae = np.mean(pixel_diff)/255
    return mae


def calculate_fbeta(image1, image2):
    pred = (image1>0).astype(int)
    true = image2//255
    beta = 0.3
    fbeta_list = []
    for t in range(255):
        pred = (image1>t).astype(int)
        fbeta = fbeta_score(np.reshape(true, (-1)), np.reshape(pred, (-1)), beta=beta)
        fbeta_list.append(fbeta)
    return fbeta_list


# res_dir = './dataset/test/LFSD/result_modify_2/'
res_dir = './icraft/result/LFSD/images_16_kld_modify_2/'
gt_dir = './dataset/test/LFSD/GT/'

res_list = sorted(os.listdir(res_dir))
gt_list = sorted(os.listdir(gt_dir))
# res_path = res_dir + '1_JLDCF.png'
# gt_path = gt_dir + '1_GT.png'
# res = cv2.imread(res_path, 1)
# gt = cv2.imread(gt_path, 1)
# print(res.shape, gt.shape)
# print(np.max(res))
# print(np.max(gt))
# print(calculate_ssim(res, gt))
# print(calculate_mae(res, gt))


SSIM = []
mae = []
fbeta = []
for idx in tqdm(range(len(res_list))):
    # print(res_list[idx], gt_list[idx])
    # res_path = res_dir + res_list[idx]
    # gt_path = gt_dir + gt_list[idx]
    res_path = res_dir + '{}.png'.format(idx+1)
    gt_path = gt_dir + '{}_GT.png'.format(idx+1)
    res = cv2.imread(res_path, 0)
    gt = cv2.imread(gt_path, 0)
    SSIM.append(calculate_ssim(res, gt))
    mae.append(calculate_mae(res, gt))
    fbeta.append(calculate_fbeta(res, gt))
    # print(f)
    # print(res)
    # pred = (res>0).astype(int)
    # true = gt//255
    # print(true.shape, pred.shape)
    # print((res>0).astype(int))
    # print(res)
    # print(np.sum(gt)/255)
    # if idx == 2:
    #     break
print('images_16_kld_modify_0')
print("SSIM: ", np.mean(np.array(SSIM)))
print("mae: ", np.mean(np.array(mae)))
print("fbeta: ", np.max(np.mean(fbeta, axis=0)))
# print(np.array(fbeta).shape)
# print(np.max(np.mean(fbeta, axis=0)))