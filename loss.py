import numpy as np
import tensorflow as tf
from model import yolo_head


def box_iou(box1, box2):
    """
    calculate iou
    :param box1: tensor, shape=[grid_h, grid_w, anchors, xywh]
    :param box2: tensor, shape=[box_num, xywh]
    :return: iou: tensor, shape=[grid_h, grid_w, anchors, box_num]
    """

    box1 = tf.expand_dims(box1, -2)  # shape=(grid_h, grid_w, num_anchors, 1, 4)
    box1_xy = box1[..., :2]
    box1_wh = box1[..., 2:4]
    box1_mins = box1_xy - box1_wh / 2.
    box1_maxs = box1_xy + box1_wh / 2.

    box2 = tf.expand_dims(box2, 0)  # shape=(1, num_true_boxes, 4)
    box2_xy = box2[..., :2]
    box2_wh = box2[..., 2:4]
    box2_mins = box2_xy - box2_wh / 2.
    box2_maxs = box2_xy + box2_wh / 2.

    intersect_mins = tf.maximum(box1_mins, box2_mins)  # shape=(grid_h, grid_w, num_anchors, num_true_boxes, 2)
    intersect_maxs = tf.minimum(box1_maxs, box2_maxs)
    intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    box1_area = box1_wh[..., 0] * box1_wh[..., 1]
    box2_area = box2_wh[..., 0] * box2_wh[..., 1]
    iou = intersect_area / (
            box1_area + box2_area - intersect_area)  # shape=(grid_h, grid_w, num_anchors, num_true_boxes)
    return iou


def yolo_loss(yolo_output, y_true, anchors, num_classes, ignore_thresh=.5):
    """
    Calculate loss for yolo
    :param yolo_output: list of tensor has shape=(m, grid_h, grid_w, num_anchors, attrs)
    :param y_true: list of array has shape=(m, grid_h, grid_w, num_anchor, attrs)
    :param anchors: array, (N, 2)
    :param num_classes: integer
    :param ignore_thresh: float, the iou threshold whether to ignore object confidence loss
    :return: loss: tensor, (1,)
    """
    num_layers = len(anchors) // 3
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    input_shape = tf.cast(tf.shape(yolo_output[0])[1:3] * 32, tf.float32)
    grid_shapes = [tf.cast(yolo_output[l].get_shape().as_list()[1:3], tf.float32) for l in range(num_layers)]

    loss = 0
    m = tf.shape(yolo_output[0])[0]
    mf = tf.cast(m, dtype=tf.float32)

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]  # shape(m, grid_h, grid_w, num_anchors, 1)
        object_mask_bool = tf.cast(object_mask, tf.bool)
        object_mask_concat = tf.concat([object_mask_bool, object_mask_bool], axis=-1)
        true_class_probs = y_true[l][..., 5]  # shape(m, grid_h, grid_w, num_anchors, num_classes)

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_output[l], anchors[anchor_mask[l]], num_classes, input_shape,
                                                     calc_loss=True)
        pred_box = tf.concat([pred_xy, pred_wh], axis=-1)  # shape=(m, grid_h, grid_w, num_anchors, 4)

        # Darknet raw box to calculate loss
        raw_true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid
        raw_true_wh = tf.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = tf.where(object_mask_concat, raw_true_wh, tf.zeros_like(raw_true_wh))
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

        # Find ignore mask, iterate over each of batch
        ignore_mask = tf.TensorArray(tf.float32, size=1, dynamic_size=True)

        def loop_body(b, ignore_mask):
            print(l, b)
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])  # shape=(t,4)
            iou = box_iou(pred_box[b], true_box)  # shape=(grid_h, grid_w, num_anchor, num_true_box)
            best_iou = tf.reduce_max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, tf.cast(best_iou < ignore_thresh, tf.float32))
            return b + 1, ignore_mask

        _, ignore_mask = tf.while_loop(lambda b, ignore_mask: b < m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = tf.expand_dims(ignore_mask, -1)  # shape=(m, grid_h, grid_w, num_anchor, 1)

        xy_loss = object_mask * box_loss_scale * tf.nn.sigmoid_cross_entropy_with_logits(labels=raw_true_xy,
                                                                                         logits=raw_pred[..., 0:2])
        wh_loss = object_mask * box_loss_scale * 0.5 * tf.square(raw_true_wh - raw_pred[..., 2:4])
        confidence_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                                logits=raw_pred[..., 4:5]) + \
                          (1 - object_mask) * \
                          tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                  logits=raw_pred[..., 4:5]) * ignore_mask

        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=true_class_probs,
                                                                           logits=raw_pred[..., 5:])

        xy_loss = tf.reduce_sum(xy_loss) / mf
        wh_loss = tf.reduce_sum(wh_loss) / mf
        confidence_loss = tf.reduce_sum(confidence_loss) / mf
        class_loss = tf.reduce_sum(class_loss) / mf

        loss += xy_loss + wh_loss + confidence_loss + class_loss

    return loss


