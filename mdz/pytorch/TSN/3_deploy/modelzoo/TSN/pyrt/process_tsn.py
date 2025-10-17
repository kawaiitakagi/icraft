
import numpy as np
from collections import OrderedDict
import copy
from accuracy import *
def read_list_from_txt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


def load_data_list(lines):
    data_list = []
    for line in lines:
        line_split = line.strip().split(" ")
        filename, label = line_split
        label = int(label)
        data_list.append(dict(filename=filename, label=label))
    return data_list

def compute_metrics(results):
        """Compute the metrics from processed results.

        Args:
            results (list[dict{"pred":score,"label":gt_label}]): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        labels = [x['label'] for x in results]

        eval_results = dict()
        # Ad-hoc for RGBPoseConv3D
        if isinstance(results[0]['pred'], dict):

            for item_name in results[0]['pred'].keys():
                preds = [x['pred'][item_name] for x in results]
                eval_result = calculate(preds, labels)
                eval_results.update(
                    {f'{item_name}_{k}': v
                     for k, v in eval_result.items()})

            if len(results[0]['pred']) == 2 and \
                    'rgb' in results[0]['pred'] and \
                    'pose' in results[0]['pred']:

                rgb = [x['pred']['rgb'] for x in results]
                pose = [x['pred']['pose'] for x in results]

                preds = {
                    '1:1': get_weighted_score([rgb, pose], [1, 1]),
                    '2:1': get_weighted_score([rgb, pose], [2, 1]),
                    '1:2': get_weighted_score([rgb, pose], [1, 2])
                }
                for k in preds:
                    eval_result = calculate(preds[k], labels)
                    eval_results.update({
                        f'RGBPose_{k}_{key}': v
                        for key, v in eval_result.items()
                    })
            return eval_results

        # Simple Acc Calculation
        else:
            preds = [x['pred'] for x in results]
            return calculate(preds, labels)
        
def calculate(preds, labels):
    """Compute the metrics from processed results.

    Args:
        preds (list[np.ndarray]): List of the prediction scores.
        labels (list[int | np.ndarray]): List of the labels.

    Returns:
        dict: The computed metrics. The keys are the names of the metrics,
        and the values are corresponding results.
    """
    eval_results = OrderedDict()
    # metric_options = copy.deepcopy(metric_options)
    metrics = ('top_k_accuracy', 'mean_class_accuracy')
    for metric in metrics:
        if metric == 'top_k_accuracy':
            # topk = metric_options.setdefault('top_k_accuracy',
            #                                     {}).setdefault(
            #                                         'topk', (1, 5))
            topk = (1,5)

            if not isinstance(topk, (int, tuple)):
                raise TypeError('topk must be int or tuple of int, '
                                f'but got {type(topk)}')

            if isinstance(topk, int):
                topk = (topk, )

            top_k_acc = top_k_accuracy(preds, labels, topk)
            for k, acc in zip(topk, top_k_acc):
                eval_results[f'top{k}'] = acc

        if metric == 'mean_class_accuracy':
            mean1 = mean_class_accuracy(preds, labels)
            eval_results['mean1'] = mean1

        if metric in [
                'mean_average_precision',
                'mmit_mean_average_precision',
        ]:
            if metric == 'mean_average_precision':
                mAP = mean_average_precision(preds, labels)
                eval_results['mean_average_precision'] = mAP

            elif metric == 'mmit_mean_average_precision':
                mAP = mmit_mean_average_precision(preds, labels)
                eval_results['mmit_mean_average_precision'] = mAP

    return eval_results
       
# def get_weighted_score(score_list, coeff_list):
#     """Get weighted score with given scores and coefficients.

#     Given n predictions by different classifier: [score_1, score_2, ...,
#     score_n] (score_list) and their coefficients: [coeff_1, coeff_2, ...,
#     coeff_n] (coeff_list), return weighted score: weighted_score =
#     score_1 * coeff_1 + score_2 * coeff_2 + ... + score_n * coeff_n

#     Args:
#         score_list (list[list[np.ndarray]]): List of list of scores, with shape
#             n(number of predictions) X num_samples X num_classes
#         coeff_list (list[float]): List of coefficients, with shape n.

#     Returns:
#         list[np.ndarray]: List of weighted scores.
#     """
#     assert len(score_list) == len(coeff_list)
#     num_samples = len(score_list[0])
#     for i in range(1, len(score_list)):
#         assert len(score_list[i]) == num_samples

#     scores = np.array(score_list)  # (num_coeff, num_samples, num_classes)
#     coeff = np.array(coeff_list)  # (num_coeff, )
#     weighted_scores = list(np.dot(scores.T, coeff).T)
#     return weighted_scores

# def top_k_accuracy(scores, labels, topk=(1, )):
#     """Calculate top k accuracy score.

#     Args:
#         scores (list[np.ndarray]): Prediction scores for each class.
#         labels (list[int]): Ground truth labels.
#         topk (tuple[int]): K value for top_k_accuracy. Default: (1, ).

#     Returns:
#         list[float]: Top k accuracy score for each k.
#     """
#     res = []
#     labels = np.array(labels)[:, np.newaxis]
#     for k in topk:
#         max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
#         match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
#         topk_acc_score = match_array.sum() / match_array.shape[0]
#         res.append(topk_acc_score)

#     return res

# def mean_class_accuracy(scores, labels):
#     """Calculate mean class accuracy.

#     Args:
#         scores (list[np.ndarray]): Prediction scores for each class.
#         labels (list[int]): Ground truth labels.

#     Returns:
#         np.ndarray: Mean class accuracy.
#     """
#     pred = np.argmax(scores, axis=1)
#     cf_mat = confusion_matrix(pred, labels).astype(float)

#     cls_cnt = cf_mat.sum(axis=1)
#     cls_hit = np.diag(cf_mat)

#     mean_class_acc = np.mean(
#         [hit / cnt if cnt else 0.0 for cnt, hit in zip(cls_cnt, cls_hit)])

#     return mean_class_acc

# def mean_average_precision(scores, labels):
#     """Mean average precision for multi-label recognition.

#     Args:
#         scores (list[np.ndarray]): Prediction scores of different classes for
#             each sample.
#         labels (list[np.ndarray]): Ground truth many-hot vector for each
#             sample.

#     Returns:
#         np.float64: The mean average precision.
#     """
#     results = []
#     scores = np.stack(scores).T
#     labels = np.stack(labels).T

#     for score, label in zip(scores, labels):
#         precision, recall, _ = binary_precision_recall_curve(score, label)
#         ap = -np.sum(np.diff(recall) * np.array(precision)[:-1])
#         results.append(ap)
#     results = [x for x in results if not np.isnan(x)]
#     if results == []:
#         return np.nan
#     return np.mean(results)

# def binary_precision_recall_curve(y_score, y_true):
#     """Calculate the binary precision recall curve at step thresholds.

#     Args:
#         y_score (np.ndarray): Prediction scores for each class.
#             Shape should be (num_classes, ).
#         y_true (np.ndarray): Ground truth many-hot vector.
#             Shape should be (num_classes, ).

#     Returns:
#         precision (np.ndarray): The precision of different thresholds.
#         recall (np.ndarray): The recall of different thresholds.
#         thresholds (np.ndarray): Different thresholds at which precision and
#             recall are tested.
#     """
#     assert isinstance(y_score, np.ndarray)
#     assert isinstance(y_true, np.ndarray)
#     assert y_score.shape == y_true.shape

#     # make y_true a boolean vector
#     y_true = (y_true == 1)
#     # sort scores and corresponding truth values
#     desc_score_indices = np.argsort(y_score, kind='mergesort')[::-1]
#     y_score = y_score[desc_score_indices]
#     y_true = y_true[desc_score_indices]
#     # There may be ties in values, therefore find the `distinct_value_inds`
#     distinct_value_inds = np.where(np.diff(y_score))[0]
#     threshold_inds = np.r_[distinct_value_inds, y_true.size - 1]
#     # accumulate the true positives with decreasing threshold
#     tps = np.cumsum(y_true)[threshold_inds]
#     fps = 1 + threshold_inds - tps
#     thresholds = y_score[threshold_inds]

#     precision = tps / (tps + fps)
#     precision[np.isnan(precision)] = 0
#     recall = tps / tps[-1]
#     # stop when full recall attained
#     # and reverse the outputs so recall is decreasing
#     last_ind = tps.searchsorted(tps[-1])
#     sl = slice(last_ind, None, -1)

#     return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]

# def confusion_matrix(y_pred, y_real, normalize=None):
#     """Compute confusion matrix.

#     Args:
#         y_pred (list[int] | np.ndarray[int]): Prediction labels.
#         y_real (list[int] | np.ndarray[int]): Ground truth labels.
#         normalize (str | None): Normalizes confusion matrix over the true
#             (rows), predicted (columns) conditions or all the population.
#             If None, confusion matrix will not be normalized. Options are
#             "true", "pred", "all", None. Default: None.

#     Returns:
#         np.ndarray: Confusion matrix.
#     """
#     if normalize not in ['true', 'pred', 'all', None]:
#         raise ValueError("normalize must be one of {'true', 'pred', "
#                          "'all', None}")

#     if isinstance(y_pred, list):
#         y_pred = np.array(y_pred)
#         if y_pred.dtype == np.int32:
#             y_pred = y_pred.astype(np.int64)
#     if not isinstance(y_pred, np.ndarray):
#         raise TypeError(
#             f'y_pred must be list or np.ndarray, but got {type(y_pred)}')
#     if not y_pred.dtype == np.int64:
#         raise TypeError(
#             f'y_pred dtype must be np.int64, but got {y_pred.dtype}')

#     if isinstance(y_real, list):
#         y_real = np.array(y_real)
#         if y_real.dtype == np.int32:
#             y_real = y_real.astype(np.int64)
#     if not isinstance(y_real, np.ndarray):
#         raise TypeError(
#             f'y_real must be list or np.ndarray, but got {type(y_real)}')
#     if not y_real.dtype == np.int64:
#         raise TypeError(
#             f'y_real dtype must be np.int64, but got {y_real.dtype}')

#     label_set = np.unique(np.concatenate((y_pred, y_real)))
#     num_labels = len(label_set)
#     max_label = label_set[-1]
#     label_map = np.zeros(max_label + 1, dtype=np.int64)
#     for i, label in enumerate(label_set):
#         label_map[label] = i

#     y_pred_mapped = label_map[y_pred]
#     y_real_mapped = label_map[y_real]

#     confusion_mat = np.bincount(
#         num_labels * y_real_mapped + y_pred_mapped,
#         minlength=num_labels**2).reshape(num_labels, num_labels)

#     with np.errstate(all='ignore'):
#         if normalize == 'true':
#             confusion_mat = (
#                 confusion_mat / confusion_mat.sum(axis=1, keepdims=True))
#         elif normalize == 'pred':
#             confusion_mat = (
#                 confusion_mat / confusion_mat.sum(axis=0, keepdims=True))
#         elif normalize == 'all':
#             confusion_mat = (confusion_mat / confusion_mat.sum())
#         confusion_mat = np.nan_to_num(confusion_mat)

#     return confusion_mat