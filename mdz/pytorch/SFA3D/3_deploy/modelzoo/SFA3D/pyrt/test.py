import kitti_common as kitti
from eval import get_official_eval_result, get_coco_eval_result


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


label_split_file = './dataset/kitti/ImageSets/val.txt'
val_image_ids = _read_imageset_file(label_split_file)
label_path = './dataset/kitti/training/label_2/'
gt_annos = kitti.get_label_annos(label_path, val_image_ids)

# det_path = './results/eval/'
det_path = './icraft/result/BY_mix_null_txt/'
dt_annos = kitti.get_label_annos(det_path, val_image_ids)
# print(dt_annos)
print(get_official_eval_result(gt_annos, dt_annos, 0))
# print('!!')