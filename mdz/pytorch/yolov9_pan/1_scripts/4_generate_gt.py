import numpy as np
from pycocotools.coco import COCO
import cv2
import sys
import os
sys.path.append(R"../0_yolov9")
from utils.plots import Annotator, colors
from utils.coco_utils import getCocoIds, getMappingId, getMappingIndex
from yolov9_utils import panoptic_merge_coco

# def generate_instance_mask(coco, img_id):
#     img_info = coco.loadImgs(img_id)[0]
#     width, height = img_info['width'], img_info['height']
#     instance_mask = np.zeros((height, width), dtype=np.uint8)
    
#     ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
#     anns = coco.loadAnns(ann_ids)
    
#     for ann in anns:
#         instance_mask += coco.annToMask(ann) * ann['category_id']
    
#     return instance_mask

# def generate_instance_masks(coco, img_id):
#     img_info = coco.loadImgs(img_id)[0]
#     width, height = img_info['width'], img_info['height']
#     instance_masks = []
#     instance_labels = []
    
#     ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
#     anns = coco.loadAnns(ann_ids)
    
#     for ann in anns:
#         mask = coco.annToMask(ann)
#         instance_masks.append(mask)
#         instance_labels.append(ann['category_id'])
    
#     instance_masks = np.array(instance_masks)
#     instance_labels = np.array(instance_labels)
    
#     return instance_masks, instance_labels


def generate_instance_masks(coco, img_id):
    img_info = coco.loadImgs(img_id)[0]
    width, height = img_info['width'], img_info['height']
    instance_masks = []
    instance_labels = []
    instance_areas = []  # 用于记录实例的面积
    
    ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    
    for ann in anns:
        mask = coco.annToMask(ann)
        instance_masks.append(mask)
        instance_labels.append(ann['category_id'])
        
        # 计算实例的面积并保存
        area = np.sum(mask)
        instance_areas.append(area)
    
    # 将实例掩码、标签和面积转换为数组
    instance_masks = np.array(instance_masks)
    instance_labels = np.array(instance_labels)
    instance_areas = np.array(instance_areas)
    
    # 根据面积对掩码和标签进行排序
    sorted_indices = np.argsort(instance_areas)[::-1]
    instance_masks = instance_masks[sorted_indices]
    instance_labels = instance_labels[sorted_indices]
    
    return instance_masks, instance_labels

def generate_stuff_mask(coco, img_id):
    img_info = coco.loadImgs(img_id)[0]
    width, height = img_info['width'], img_info['height']
    stuff_mask = np.zeros((height, width), dtype=np.uint8)
    
    ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    
    for ann in anns:
        if ann['category_id'] >= 92:  # Stuff category ids in COCO start from 92
            stuff_mask += coco.annToMask(ann)*ann['category_id']
    
    return stuff_mask

# Load COCO annotations
coco_instance = COCO('../instances_val2017.json')
coco_stuff = COCO('../stuff_val2017.json')

img_ids_instance = set(coco_instance.getImgIds())
img_ids_stuff = set(coco_stuff.getImgIds())
img_ids = list(img_ids_instance.intersection(img_ids_stuff))

fakelist = []
for img_id in img_ids:
    instance_masks, instance_labels = generate_instance_masks(coco_instance, img_id)
    stuff_mask = generate_stuff_mask(coco_stuff, img_id)
    panoptic = panoptic_merge_coco(stuff_mask, instance_masks, instance_labels, min_area=0)
    # print(instance_labels)
    # print(np.unique(panoptic[:,:,1]))
    if (len(np.unique(panoptic[:,:,1]))-1) != len(instance_labels):
         fakelist.append(img_id)
    panoptic[:, :, 1] = panoptic[:, :, 1] // 1000
    panoptic[:, :, 2] = panoptic[:, :, 2] % 256
    panoptic = panoptic.astype('uint8')
    img_name = f'{img_id}' + ".png"
    os.makedirs("../1_scripts/coco_val/", exist_ok=True)
    file_path = os.path.join("../1_scripts/coco_val/",img_name)
    cv2.imwrite(file_path, panoptic)

print(fakelist)
print(len(fakelist))





# # Generate masks for a specific image
# img_id = 2157
# instance_masks, instance_labels = generate_instance_masks(coco_instance, img_id)
# stuff_mask = generate_stuff_mask(coco_stuff, img_id)
# print(instance_labels)

# panoptic = panoptic_merge_coco(stuff_mask, instance_masks, instance_labels, min_area=0)
# print("semask: ", np.unique(stuff_mask))
# print("panoptic: ", np.unique(panoptic[:, :, 2]))
# # panoptic[:, :, 1] = panoptic[:, :, 1] // 1000
# # panoptic[:, :, 2] = panoptic[:, :, 2] % 256
# # panoptic = panoptic.astype('uint8')

# # for image show
# color_image = np.zeros(panoptic.shape, dtype=np.uint8)
# for y in range(panoptic.shape[0]):
#     for x in range(panoptic.shape[1]):
#         semantic_id = panoptic[y, x, 2]
#         if semantic_id != 0:
#             color = colors((semantic_id // 1000) + (semantic_id % 1000))
#         else:
#             color = (0, 0, 0)  # black
#         color_image[y, x] = color
# cv2.imshow('Panoptic Image', color_image)
# cv2.waitKey()



