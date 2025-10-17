import PIL.Image as Image
import numpy as np
from collections import defaultdict
import os
from tqdm import tqdm
import cv2
# RGB通道会有变换，所以0通道为画图用，1通道只有实例，2通道为计算用
# img = Image.open(pred_path)
# img.show()


class PQStatCat():
        def __init__(self):
            self.iou = 0.0
            self.tp = 0
            self.fp = 0
            self.fn = 0

        def __iadd__(self, pq_stat_cat):
            self.iou += pq_stat_cat.iou 
            self.tp += pq_stat_cat.tp
            self.fp += pq_stat_cat.fp
            self.fn += pq_stat_cat.fn
            return self

class PQStat():
    def __init__(self):
        self.pq_per_cat = defaultdict(PQStatCat)

    def __getitem__(self, i):
        return self.pq_per_cat[i]

    def __iadd__(self, pq_stat):
        for label, pq_stat_cat in pq_stat.pq_per_cat.items():
            self.pq_per_cat[label] += pq_stat_cat
        return self

    def pq_average(self, categories, isthing):
        pq, sq, rq, n = 0, 0, 0, 0
        per_class_results = {}
        for label in categories:
            if label == 0:
                continue
            if isthing is not None:
                cat_isthing = (label < 92)
                if isthing != cat_isthing:
                    continue
            iou = self.pq_per_cat[label].iou
            tp = self.pq_per_cat[label].tp
            fp = self.pq_per_cat[label].fp
            fn = self.pq_per_cat[label].fn
            if tp + fp + fn == 0:
                per_class_results[label] = {'pq': 0.0, 'sq': 0.0, 'rq': 0.0}
                continue
            n += 1
            pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
            sq_class = iou  / tp if tp != 0 else 0
            rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
            per_class_results[label] = {'pq': pq_class, 'sq': sq_class, 'rq': rq_class}
            pq += pq_class
            sq += sq_class
            rq += rq_class

        return {'pq': pq / n, 'sq': sq / n, 'rq': rq / n, 'n': n}, per_class_results

VOID = 0
pq_stat = PQStat()

def valuation(pred_path, gt_path):
    pred_segms = {}
    pan_gt = np.array(Image.open(gt_path), dtype=np.uint8)[:,:,2]
    pan_pred = np.array(Image.open(pred_path), dtype=np.uint8)[:,:,2]
    # pan_gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)[:,:,0]
    # pan_pred = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)[:,:,0]

    labels_pred, labels_cnt_pred = np.unique(pan_pred,return_counts=True)
    for label, label_cnt in zip(labels_pred, labels_cnt_pred):
        pred_segms[label] = {}  
        pred_segms[label]['area'] = label_cnt
        pred_segms[label]['category_id'] = label
    # print(pred_segms)

    gt_segms = {}
    labels_gt, labels_cnt_gt = np.unique(pan_gt,return_counts=True)
    # print(labels_gt)
    for label, label_cnt in zip(labels_gt, labels_cnt_gt):
        gt_segms[label] = {}  
        gt_segms[label]['area'] = label_cnt
        gt_segms[label]['category_id'] = label
    # print(gt_segms)

    pan_gt_pred = pan_gt * 256 + pan_pred
    gt_pred_map = {}
    labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
    # print(labels)
    # print(labels_cnt)

    for label, intersection in zip(labels, labels_cnt):
        gt_id = label // 256
        pred_id = label % 256
        gt_pred_map[(gt_id, pred_id)] = intersection
    # print(gt_pred_map)

    # count all matched pairs
    gt_matched = set()
    pred_matched = set()
    for label_tuple, intersection in gt_pred_map.items():
        gt_label, pred_label = label_tuple

        if gt_label not in gt_segms:
            continue
        if pred_label not in pred_segms:
            continue
        if gt_segms[gt_label]['category_id'] != pred_segms[pred_label]['category_id']:
            continue

        union = pred_segms[pred_label]['area'] + gt_segms[gt_label]['area'] - intersection - gt_pred_map.get((VOID, pred_label), 0)
        iou = intersection / union
        if iou > 0.5:
            pq_stat[gt_segms[gt_label]['category_id']].tp += 1
            pq_stat[gt_segms[gt_label]['category_id']].iou += iou 
            gt_matched.add(gt_label)
            pred_matched.add(pred_label)

    # count false positives
    for gt_label, gt_info in gt_segms.items():
        if gt_label in gt_matched:
            continue
        pq_stat[gt_info['category_id']].fn += 1

    # count false positives
    for pred_label, pred_info in pred_segms.items():
        if pred_label in pred_matched:
            continue
        # intersection of the segment with VOID
        intersection = gt_pred_map.get((VOID, pred_label), 0)
        # predicted segment is ignored if more than half of the segment correspond to VOID and CROWD regions
        if intersection / pred_info['area'] > 0.5:
            continue
        pq_stat[pred_info['category_id']].fp += 1
    return pq_stat

all_instances_ids = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 27, 28,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
    61, 62, 63, 64, 65, 67, 70,
    72, 73, 74, 75, 76, 77, 78, 79, 80,
    81, 82, 84, 85, 86, 87, 88, 89, 90,
]

all_stuff_ids = [
    92, 93, 94, 95, 96, 97, 98, 99, 100,
    101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
    111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
    121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
    131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
    141, 142, 143, 144, 145, 146, 147, 148, 149, 150,
    151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
    161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
    171, 172, 173, 174, 175, 176, 177, 178, 179, 180,
    181, 182,
    # other
    183,
    # unlabeled
    0,
]  
labels_gt = all_instances_ids + all_stuff_ids

def get_corresponding_pairs(pred_path, gt_path):
    pred_files = [filename for filename in os.listdir(pred_path) 
                  if filename.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    
    pairs = []

    for pred_file in pred_files:
        # 提取预测文件的编号
        pred_file_number = pred_file.split('.')[0]
        gt_filename = pred_file_number.lstrip('0') + '.png'  # 去掉前导零并添加 .png 后缀
        gt_file = os.path.join(gt_path, gt_filename)
        
        # 检查 gt_file 是否存在于 gt_path 中
        if os.path.exists(gt_file):
            # 添加 (pred_file, gt_file) 对到列表中
            pairs.append((os.path.join(pred_path, pred_file), gt_file))

    return pairs

def calculate_and_accumulate(pq_stat, results, labels_gt):
    metrics = [("All", None), ("Things", True), ("Stuff", False)]
    for name, isthing in metrics:
        res, _ = pq_stat.pq_average(labels_gt, isthing=isthing)
        if name not in results:
            results[name] = {'pq': 0, 'sq': 0, 'rq': 0, 'n': 0}
        results[name]['pq'] += res['pq'] * res['n']
        results[name]['sq'] += res['sq'] * res['n']
        results[name]['rq'] += res['rq'] * res['n']
        results[name]['n'] += res['n']

# pred_path = '../1_scripts/val_test'
pred_path = '../3_deploy/modelzoo/yolov9_pan/io/output_16'
gt_path = '../1_scripts/coco_val'
corresponding_pairs = get_corresponding_pairs(pred_path, gt_path)

for pair in tqdm(corresponding_pairs[:500], desc="loading"):
    pq_stat = valuation(pair[0], pair[1])
    pq_stat += pq_stat

metrics = [("All", None), ("Things", True), ("Stuff", False)]
results = {}

# labels_gt = [0  ,1 ,35 ,120 ,159]
for name, isthing in metrics:
    results[name], per_class_results = pq_stat.pq_average(labels_gt, isthing=isthing)
    if name == 'All':
        results['per_class'] = per_class_results

# print(results['All']['pq'])

metrics = ["All", "Things", "Stuff"]
print("-" * 41)
print("{:14s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "PQ", "SQ", "RQ", "N"))
for name in metrics:
    print("{:14s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}".format(
        name,
        100 * results[name]['pq'],
        100 * results[name]['sq'],
        100 * results[name]['rq'],
        results[name]['n']
    ))


# results = {}
# batch_size = 1
# for i in tqdm(range(0, len(corresponding_pairs[:10]), batch_size), desc="Processing in batches"):
#     pq_stat = PQStat()
#     batch_pairs = corresponding_pairs[i:i + batch_size]
#     for pair in batch_pairs:
#         pq_stat += valuation(pair[0], pair[1])
#     calculate_and_accumulate(pq_stat, results, labels_gt)
# print(results)
# final_results = {}
# for name, res in results.items():
#     final_results[name] = {
#         'pq': res['pq'] / res['n'],
#         'sq': res['sq'] / res['n'],
#         'rq': res['rq'] / res['n'],
#         'n': res['n']
#     }
# print(final_results)