import numpy as np


def box_iou_numpy(box1, box2):
    """
    calculate iou
    :param box1: matrix, shape=[..., 4], dimension: N
    :param box2: matrix, shape=[..., 4], dimension: K
    :return: iou: matrix, has dimension: (N-1) + (K-1)
    """

    box1 = np.expand_dims(box1, -2)  # shape=(grid_h, grid_w, num_anchors, 1, 4)
    box1_mins = box1[..., 0:2]
    box1_maxs = box1[..., 2:4]
    box1_wh = box1_maxs - box1_mins

    box2 = np.expand_dims(box2, 0)  # shape=(1, num_true_boxes, 4)
    box2_mins = box2[..., 0:2]
    box2_maxs = box2[..., 2:4]
    box2_wh = box2_maxs - box2_mins

    intersect_mins = np.maximum(box1_mins, box2_mins)  # shape=(grid_h, grid_w, num_anchors, num_true_boxes, 2)
    intersect_maxs = np.minimum(box1_maxs, box2_maxs)
    intersect_wh = np.maximum(intersect_maxs - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    box1_area = box1_wh[..., 0] * box1_wh[..., 1]
    box2_area = box2_wh[..., 0] * box2_wh[..., 1]
    iou = intersect_area / (
            box1_area + box2_area - intersect_area)  # shape=(grid_h, grid_w, num_anchors, num_true_boxes)
    #     print(iou)
    return iou
