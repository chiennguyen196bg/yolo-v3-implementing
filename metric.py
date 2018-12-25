import numpy as np
from util import box_iou_numpy


def _process_predict_an_image(predict, ground_truth, num_classes, iou_threshold):
    """
    Calculate predict_box_result has shape (n, 3): class, confidence, is true positive
    param predict: shape=(max_boxes, 6)
    param ground_truth: shape=(max_boxes, 5)
    """
    predict = predict[predict[:, 4] > 0]  # shape=(n, 6) filter predicts that have score > 0
    ground_truth = ground_truth[ground_truth[:, 2] > 0]  # shape=(m, 5) filter ground truths that have xmax > 0
    predict_box_results_total = []
    #     print(ground_truth)
    for c in range(num_classes):
        class_predict = predict[predict[:, 5] == c]  # filter predicts for class c
        class_ground_truth = ground_truth[ground_truth[:, 4] == c]  # filter ground truths for class c
        class_predict_boxes = class_predict[:, 0:4]
        class_ground_truth_boxes = class_ground_truth[:, 0:4]
        n_predict_boxes = class_predict_boxes.shape[0]
        n_grouth_truth_boxes = class_ground_truth_boxes.shape[0]

        if n_predict_boxes == 0:
            continue
        # make placeholder for predict_box_result
        predict_box_results = np.zeros((n_predict_boxes, 3), dtype=np.float32)  # class, confidence, true_positive
        predict_box_results[:, 0] = c
        predict_box_results[:, 1] = class_predict[:, 4]

        if n_grouth_truth_boxes == 0:
            predict_box_results_total.append(predict_box_results)
            continue

        iou = box_iou_numpy(class_predict_boxes, class_ground_truth_boxes)
        # convert iou to (predict_box_index, ground_truth_index, iou_value)
        i = np.tile(np.reshape(np.arange(0, n_predict_boxes), [-1, 1, 1]),
                    [1, n_grouth_truth_boxes, 1])  # index for predict_box
        j = np.tile(np.reshape(np.arange(0, n_grouth_truth_boxes), [1, -1, 1]),
                    [n_predict_boxes, 1, 1])  # index for ground truth box
        grid = np.concatenate([i, j], axis=-1).reshape(-1, 2)
        iou_flatten = iou.flatten()
        sort_key = iou_flatten.argsort()[::-1]
        iou_flatten = iou_flatten[sort_key]
        iou_mask = iou_flatten > iou_threshold

        indexes_sorted_by_iou = grid[sort_key]
        indexes_sorted_by_iou = indexes_sorted_by_iou[iou_mask]

        ground_truth_used_flag = [False] * n_grouth_truth_boxes
        predict_used_flag = [False] * n_predict_boxes

        for index_pair in indexes_sorted_by_iou:
            p, g = index_pair
            if not ground_truth_used_flag[g] and not predict_used_flag[p]:
                ground_truth_used_flag[g] = predict_used_flag[p] = True
                predict_box_results[p, 2] = 1.0
        predict_box_results_total.append(predict_box_results)
    if len(predict_box_results_total) > 0:
        predict_box_results_total = np.concatenate(predict_box_results_total, axis=0)
    return predict_box_results_total


def _voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
        mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    # print(rec)
    # print(prec)
    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def cal_AP(predict, ground_truth, num_classes, iou_threshold):
    """
    param predict: shape=(m, max_boxes, 6)
    param ground_truth: shape=(m, max_boxes, 5)
    param num_classes
    """
    batch_size = predict.shape[0]
    predict_box_result = []
    all_ground_truth_boxes = ground_truth.reshape(-1, 5)
    all_ground_truth_boxes = all_ground_truth_boxes[all_ground_truth_boxes[:, 2] > 0]
    for b in range(batch_size):
        predict_box_result_for_an_image = _process_predict_an_image(predict[b], ground_truth[b], num_classes,
                                                                    iou_threshold)
        if len(predict_box_result_for_an_image) > 0:
            predict_box_result.append(predict_box_result_for_an_image)
    if len(predict_box_result) == 0:
        return [0.] * num_classes
    predict_box_result = np.concatenate(predict_box_result, axis=0)
    AP_total = [0] * num_classes
    for c in range(num_classes):
        class_predict_box_result = predict_box_result[predict_box_result[:, 0] == c]
        class_predict_box_result = class_predict_box_result[class_predict_box_result[:, 1].argsort()[::-1]]
        # print(class_predict_box_result)
        num_class_ground_truth_boxes = len(all_ground_truth_boxes[all_ground_truth_boxes[:, 4] == c])
        acc_tp = 0
        acc_fp = 0
        precision = []
        recall = []
        for box_result in class_predict_box_result:
            if box_result[2]:
                acc_tp += 1
            else:
                acc_fp += 1
            precision.append(acc_tp / (acc_tp + acc_fp))
            recall.append(acc_tp / num_class_ground_truth_boxes)
        ap, mrc, mpre = _voc_ap(recall, precision)
        AP_total[c] = ap

    return AP_total
