import cv2
import numpy as np
import os
from tqdm import tqdm
from scipy.stats import entropy
from PIL import Image
from skimage.io import imread, imshow
from skimage.metrics import normalized_mutual_information
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def calculate_image_entropy(image_path):
    # 读取图片
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Could not open or find the image")
        return

    # 计算灰度值的概率分布
    pixel_counts = np.bincount(img.flatten())

    # 如果所有像素都相同，熵为0
    if np.all(pixel_counts == pixel_counts[0]):
        return 0

    # 计算熵
    entropy_value = entropy(pixel_counts, base=2)
    return entropy_value


def calculate_edge_intensity(image_path):
    # 读取图片
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Could not open or find the image")
        return

    # Canny边缘检测
    edges = cv2.Canny(img, low_threshold=50, high_threshold=150)

    # 计算边缘强度
    edge_intensity = np.sum(edges)
    return edge_intensity


def calculate_mutual_information(image_path1, image_path2):
    imageA = imread(image_path1, as_gray=True)
    imageB = imread(image_path2, as_gray=True)

    # 确保图像大小相同
    assert imageA.shape == imageB.shape, "Images must have the same size"

    # 计算互信息
    mi = normalized_mutual_information(imageA.flatten(), imageB.flatten())
    return mi

# def calculate_mutual_information(image_path1, image_path2):
#     # 读取图片
#     img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
#     img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
#     if img1 is None or img2 is None:
#         print("Could not open or find the images")
#         return

#     # 计算每个图像的概率分布
#     p1 = np.histogram(img1.flatten(), bins=256, range=(0, 255))[0]
#     p2 = np.histogram(img2.flatten(), bins=256, range=(0, 255))[0]

#     # 计算联合概率分布
#     joint = np.outer(p1, p2)

#     # 计算互信息
#     mi = entropy(joint, base=2) - entropy(p1, base=2) - entropy(p2, base=2)

#     return mi


def calculate_standard_deviation(image_path):
    # 读取图片
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Could not open or find the image")
        return

    # 计算标准差
    std_dev = np.std(img)

    return std_dev


def calculate_psnr(image1, image2):
    # 读取图像并转换为numpy数组
    image1 = np.array(image1)
    image2 = np.array(image2)

    # 确保图像数据类型是float
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)

    # 计算PSNR
    psnr_value = psnr(image1, image2, data_range=255)
    return psnr_value


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


# 使用图片路径
fuse_dir = 'runs/m3fd/'
# fuse_dir = './icraft/result/fuse/'
ir_dir = 'data/m3fd/ir/'
vi_dir = 'data/m3fd/vi/'
EN = []
SD = []
MI_ir = []
MI_vi = []
PSNR = []
SSIM = []
for im in tqdm(sorted(os.listdir(fuse_dir))):
    img_path = fuse_dir + im
    ir_path = ir_dir + im
    vi_path = vi_dir + im
    # EN.append(calculate_image_entropy(img_path))
    # SD.append(calculate_standard_deviation(img_path))
    # MI_ir.append(calculate_mutual_information(ir_path, img_path))
    # MI_vi.append(calculate_mutual_information(img_path, vi_path))
    image1, image2 = Image.open(img_path), Image.open(vi_path)
    PSNR.append(calculate_psnr(image1, image2))
    SSIM.append(calculate_ssim(image1, image2))
    # break
# print("Image entropy: ", np.mean(np.array(EN)))
# print("Standard Deviation: ", np.mean(np.array(SD)))
# print("mutual information of: ", np.mean(np.array(MI_ir))+np.mean(np.array(MI_vi)))
# print("mutual information of ir: ", np.mean(np.array(MI_ir)))
# print("mutual information of vi: ", np.mean(np.array(MI_vi)))
print("PSNR: ", np.mean(np.array(PSNR)))
print("SSIM: ", np.mean(np.array(SSIM)))
# image_path = 'runs/m3fd/00000.png'
# print("Image entropy: ", calculate_image_entropy(image_path))
# print("Edge intensity: ", calculate_edge_intensity(image_path))
# print("Standard Deviation: ", calculate_standard_deviation(image_path))
