import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_image_ops

# import config as cf

slim = tf.contrib.slim
tf.image.non_max_suppression = gen_image_ops.non_max_suppression_v2


@tf.contrib.framework.add_arg_scope
def _fixed_padding(inputs, kernel_size, *args, mode='CONSTANT', **kwargs):
    """
    Pads the input along the spatial dimensions independently of input size.
    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.
      data_format: The input format ('NHWC' or 'NCHW').
      mode: The mode for tf.pad.
    Returns:
      A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]], mode=mode)
    return padded_inputs


def _conv2d_fixed_padding(inputs, filters, kernel_size, strides=1):
    if strides > 1:
        inputs = _fixed_padding(inputs, kernel_size)
    inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides, padding=('SAME' if strides == 1 else 'VALID'))
    return inputs


def _darknet53_block(inputs, filters):
    shortcut = inputs
    inputs = _conv2d_fixed_padding(inputs, filters, 1)
    inputs = _conv2d_fixed_padding(inputs, filters * 2, 3)
    inputs = inputs + shortcut
    return inputs


def darknet53(inputs):
    """
    Builds Darknet-53 model
    :param inputs:
    :return:
    """
    inputs = _conv2d_fixed_padding(inputs, 32, 3)
    inputs = _conv2d_fixed_padding(inputs, 64, 3, strides=2)
    inputs = _darknet53_block(inputs, 32)
    inputs = _conv2d_fixed_padding(inputs, 128, 3, strides=2)

    for i in range(2):
        inputs = _darknet53_block(inputs, 64)

    inputs = _conv2d_fixed_padding(inputs, 256, 3, strides=2)

    for i in range(8):
        inputs = _darknet53_block(inputs, 128)

    route1 = inputs
    inputs = _conv2d_fixed_padding(inputs, 512, 3, strides=2)

    for i in range(8):
        inputs = _darknet53_block(inputs, 256)

    route2 = inputs
    inputs = _conv2d_fixed_padding(inputs, 1024, 3, strides=2)

    for i in range(4):
        inputs = _darknet53_block(inputs, 512)

    return route1, route2, inputs


def _yolo_block(inputs, filters, num_anchors, num_classes):
    inputs = _conv2d_fixed_padding(inputs, filters, 1)
    inputs = _conv2d_fixed_padding(inputs, filters * 2, 3)
    inputs = _conv2d_fixed_padding(inputs, filters, 1)
    inputs = _conv2d_fixed_padding(inputs, filters * 2, 3)
    inputs = _conv2d_fixed_padding(inputs, filters, 1)
    route = inputs
    inputs = _conv2d_fixed_padding(inputs, filters * 2, 3)
    raw_detection = slim.conv2d(inputs, num_anchors * (5 + num_classes), 1, stride=1, normalizer_fn=None,
                                activation_fn=None, biases_initializer=tf.zeros_initializer())
    return route, raw_detection


# def _detection_layer(inputs, num_classes, anchors, img_size):
#     num_anchors = len(anchors)
#     anchors_tensor = tf.constant(anchors, shape=(1, 1, 1, num_anchors, 2))
#     # feats has shape=(m, grid_height, grid_width, num_anchors * (5 + num_classes)]
#     feats = slim.conv2d(inputs, num_anchors * (5 + num_classes), 1, stride=1, normalizer_fn=None,
#                         activation_fn=None, biases_initializer=tf.zeros_initializer())
#     shape = feats.get_shape().as_list()
#     grid_size = shape[1:3]
#
#     feats = tf.reshape(-1, grid_size[0], grid_size[1], num_anchors, 5 + num_classes)
#
#     box_xy = tf.sigmoid(feats[..., 0:2])
#     # dim = grid_size[0] * grid_size[1]
#     # bbox_attrs = 5 + num_classes
#     #
#     # predictions = tf.reshape(predictions, [-1, num_anchors * dim, bbox_attrs])
#
#     # stride = (img_size[0] // grid_size[0], img_size[1] // grid_size[1])
#     #
#     # anchors = [(a[0] / stride[0], a[1] / stride[1]) for a in anchors]
#     #
#     # box_centers, box_sizes, confidence, classes = tf.split(predictions, [2, 2, 1, num_classes], axis=-1)
#     #
#     # box_centers = tf.nn.sigmoid(box_centers)
#     # confidence = tf.nn.sigmoid(confidence)
#     #
#     # grid_x = tf.range(grid_size[0], dtype=tf.float32)
#     # grid_y = tf.range(grid_size[1], dtype=tf.float32)
#     # a, b = tf.meshgrid(grid_x, grid_y)
#     #
#     # x_offset = tf.reshape(a, (-1, 1))
#     # y_offset = tf.reshape(b, (-1, 1))
#     #
#     # x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
#     # x_y_offset = tf.reshape(tf.tile(x_y_offset, [1, num_anchors]), [1, -1, 2])
#     #
#     # box_centers = box_centers + x_y_offset
#     # box_centers = box_centers * stride
#     #
#     # anchors = tf.tile(anchors, [dim, 1])
#     # box_sizes = tf.exp(box_sizes) * anchors
#     # box_sizes = box_sizes * stride
#     #
#     # detections = tf.concat([box_centers, box_sizes, confidence], axis=-1)
#     #
#     # classes = tf.nn.sigmoid(classes)
#     # predictions = tf.concat([detections, classes], axis=-1)
#     return feats


def _upsample(inputs, out_shape):
    # tf.image.resize_nearest_neighbor accepts input in format NHWC

    new_height = out_shape[1]
    new_width = out_shape[2]

    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))

    inputs = tf.identity(inputs, name='upsampled')
    return inputs


def yolo_head(raw_detect, anchors, num_classes, input_shape, calc_loss=False):
    """
    Convert
    :param raw_detect: tensor has shape=(batch_size, grid_h, grid_w, num_anchor * (5 + num_classes)
    :param anchors: array of anchor has shape=(N,2)
    :param num_classes: integer
    :param input_shape: integer, hw
    :param calc_loss: boolean, is calculating loss or not
    :return:
        grid, raw_detect, box_xy, box_wh if calculate loss
        box_xy, box_wh, box_confidence, box_class_probs if not
    """
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchor, box_params
    anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])

    grid_shape = raw_detect.get_shape().as_list()[1:3]  # height, width
    grid_y = tf.tile(tf.reshape(tf.range(grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
    grid_x = tf.tile(tf.reshape(tf.range(grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])
    grid = tf.concat([grid_x, grid_y], axis=-1)  # shape=(grid_h, grid_w, 1, 2)
    grid = tf.cast(grid, tf.float32)

    raw_detect = tf.reshape(raw_detect, (
        -1, grid_shape[0], grid_shape[1], num_anchors, 5 + num_classes))  # shape(m, grid_h, grid_w, num_anchors, attrs)

    box_xy = (tf.sigmoid(raw_detect[..., :2]) + grid) / tf.cast(grid_shape[::-1], tf.float32)
    box_wh = tf.exp(raw_detect[..., 2:4]) * anchors_tensor / tf.cast(input_shape[::-1], tf.float32)
    box_confidence = tf.sigmoid(raw_detect[..., 4:5])
    box_class_probs = tf.sigmoid(raw_detect[..., 5:])
    if calc_loss:
        return grid, raw_detect, box_xy, box_wh
    else:
        return box_xy, box_wh, box_confidence, box_class_probs


def box_iou_tensor(box1, box2):
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


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    """Process conv layer output"""
    batch_size = tf.shape(feats)[0]
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = tf.reshape(boxes, [batch_size, -1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = tf.reshape(box_scores, [batch_size, -1, num_classes])
    return boxes, box_scores


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    """Get correct boxes for calculate non max suppression"""
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = tf.cast(input_shape, dtype=tf.float32)
    image_shape = tf.cast(image_shape, dtype=tf.float32)
    new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = tf.concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)
    boxes *= tf.concat([image_shape, image_shape], axis=-1)
    return boxes


class Yolov3:
    def __init__(self, batch_norm_decay, batch_norm_epsilon, leaky_relu, anchors, num_classes):
        self.batch_norm_decay = batch_norm_decay
        self.batch_norm_epsilon = batch_norm_epsilon
        self.leaky_relu = leaky_relu
        self.anchors = np.array(anchors, dtype=np.float32).reshape(-1, 2)
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.anchors_tensor = tf.reshape(tf.constant(anchors), shape=(-1, 2))

    def yolo_inference(self, inputs, is_training=False, reuse=False, weight_decay=0.0):
        """
        Creates YOLO v3 model.
        :param inputs: a 4-D tensor of size [batch_size, height, width, channels].
            Dimension batch_size may be undefined. The channel order is RGB.
        :param num_classes: number of predicted classes.
        :param is_training: whether is training or not.
        :param reuse: whether or not the network and its variables should be reused.
        :return:
        """
        # # it will be needed later on
        # img_size = inputs.get_shape().as_list()[1:3]

        # normalize values to range [0..1]
        inputs = tf.cast(inputs, dtype=tf.float32)
        # inputs = inputs / 255

        # set batch norm params
        batch_norm_params = {
            'decay': self.batch_norm_decay,
            'epsilon': self.batch_norm_epsilon,
            'scale': True,
            'is_training': is_training,
            'fused': None,  # Use fused batch norm if possible.
        }

        # Set activation_fn and parameters for conv2d, batch_norm.
        with slim.arg_scope([slim.conv2d, slim.batch_norm, _fixed_padding], reuse=reuse):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params,
                                biases_initializer=None, weights_regularizer=slim.l2_regularizer(weight_decay),
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=self.leaky_relu)):
                with tf.variable_scope('darknet53'):
                    route_1, route_2, inputs = darknet53(inputs)

                with tf.variable_scope('yolo'):
                    route, raw_detect_1 = _yolo_block(inputs, 512, 3, self.num_classes)
                    raw_detect_1 = tf.identity(raw_detect_1, name='raw_detect_1')

                    # detect_1 = _detection_layer(inputs, num_classes, _ANCHORS[6:9], img_size)
                    # detect_1 = tf.identity(detect_1, name='detect_1')

                    inputs = _conv2d_fixed_padding(route, 256, 1)
                    upsample_size = route_2.get_shape().as_list()
                    inputs = _upsample(inputs, upsample_size)
                    inputs = tf.concat([inputs, route_2], axis=3)

                    route, raw_detect_2 = _yolo_block(inputs, 256, 3, self.num_classes)
                    raw_detect_2 = tf.identity(raw_detect_2, name='raw_detect_2')

                    # detect_2 = _detection_layer(inputs, num_classes, _ANCHORS[3:6], img_size)
                    # detect_2 = tf.identity(detect_2, name='detect_2')

                    inputs = _conv2d_fixed_padding(route, 128, 1)
                    upsample_size = route_1.get_shape().as_list()
                    inputs = _upsample(inputs, upsample_size)
                    inputs = tf.concat([inputs, route_1], axis=3)

                    _, raw_detect_3 = _yolo_block(inputs, 128, 3, self.num_classes)
                    raw_detect_3 = tf.identity(raw_detect_3, name='raw_detect_3')

                    # detect_3 = _detection_layer(inputs, num_classes, _ANCHORS[0:3], img_size)
                    # detect_3 = tf.identity(detect_3, name='detect_3')

                    # detections = tf.concat([detect_1, detect_2, detect_3], axis=1)
                    # detections = tf.identity(detections, name='detections')
                    # return detections
                    return raw_detect_1, raw_detect_2, raw_detect_3

    def yolo_loss(self, yolo_outputs, y_true, ignore_thresh=.5):
        """
        Calculate loss for yolo
        :param yolo_output: list of tensor has shape=(m, grid_h, grid_w, num_anchors, attrs)
        :param y_true: list of array has shape=(m, grid_h, grid_w, num_anchor, attrs)
        :param anchors: array, (N, 2)
        :param num_classes: integer
        :param ignore_thresh: float, the iou threshold whether to ignore object confidence loss
        :return: loss: tensor, (1,)
        """
        num_layers = self.num_anchors // 3
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
        input_shape = tf.cast(tf.shape(yolo_outputs[0])[1:3] * 32, tf.float32)
        grid_shapes = [tf.cast(yolo_outputs[l].get_shape().as_list()[1:3], tf.float32) for l in range(num_layers)]

        loss = 0
        m = tf.shape(yolo_outputs[0])[0]
        mf = tf.cast(m, dtype=tf.float32)

        for l in range(num_layers):
            object_mask = y_true[l][..., 4:5]  # shape(m, grid_h, grid_w, num_anchors, 1)
            object_mask_bool = tf.cast(object_mask, tf.bool)
            object_mask_concat = tf.concat([object_mask_bool, object_mask_bool], axis=-1)
            true_class_probs = y_true[l][..., 5:]  # shape(m, grid_h, grid_w, num_anchors, num_classes)

            grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l], self.anchors[anchor_mask[l]],
                                                         self.num_classes,
                                                         input_shape,
                                                         calc_loss=True)
            pred_box = tf.concat([pred_xy, pred_wh], axis=-1)  # shape=(m, grid_h, grid_w, num_anchors, 4)

            # Darknet raw box to calculate loss
            raw_true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid
            raw_true_wh = tf.log(y_true[l][..., 2:4] / self.anchors[anchor_mask[l]] * input_shape[::-1] + 1e-20)
            raw_true_wh = tf.where(object_mask_concat, raw_true_wh, tf.zeros_like(raw_true_wh))
            box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

            # Find ignore mask, iterate over each of batch
            ignore_mask = tf.TensorArray(tf.float32, size=1, dynamic_size=True)

            def loop_body(b, ignore_mask):
                _y_true = tf.gather(y_true[l], b)
                _object_mask_bool = tf.gather(object_mask_bool, b)
                _pred_box = tf.gather(pred_box, b)
                true_box = tf.boolean_mask(_y_true[..., 0:4], _object_mask_bool[..., 0])  # shape=(t,4)
                iou = box_iou_tensor(_pred_box, true_box)  # shape=(grid_h, grid_w, num_anchor, num_true_box)
                best_iou = tf.reduce_max(iou, axis=-1)
                ignore_mask = ignore_mask.write(b, tf.cast(best_iou < ignore_thresh, tf.float32))
                return b + 1, ignore_mask

            _, ignore_mask = tf.while_loop(lambda b, ignore_mask: b < m, loop_body, [0, ignore_mask])
            ignore_mask = ignore_mask.stack()
            ignore_mask = tf.expand_dims(ignore_mask, -1)  # shape=(m, grid_h, grid_w, num_anchor, 1)

            crood_lambda = 5.0
            noobj_lambda = 0.5

            xy_loss = crood_lambda * object_mask * box_loss_scale * \
                      tf.nn.sigmoid_cross_entropy_with_logits(labels=raw_true_xy, logits=raw_pred[..., 0:2])
            wh_loss = crood_lambda * object_mask * box_loss_scale * 0.5 * tf.square(raw_true_wh - raw_pred[..., 2:4])
            confidence_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                                    logits=raw_pred[..., 4:5]) + \
                              (1 - object_mask) * noobj_lambda * \
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

    def yolo_predict(self, yolo_outputs, image_shape, max_boxes=20, score_threshold=.6, iou_threshold=.5):
        """
        Evaluate YOLO model on given input and return filtered boxes.
        :return predict_boxes: shape=(m, max_boxes, 6). xmin, ymin, xmax, ymax, confidence, class_id
        """
        num_layers = len(yolo_outputs)
        batch_size = tf.shape(yolo_outputs[0])[0]
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
        input_shape = tf.shape(yolo_outputs[0])[1:3] * 32
        boxes = []
        box_scores = []
        for l in range(num_layers):
            _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l], self.anchors[anchor_mask[l]], self.num_classes,
                                                        input_shape, image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        boxes = tf.concat(boxes, axis=1)  # shape=(batch_size, m, 4)
        box_scores = tf.concat(box_scores, axis=1)  # shape=(batch_size, m, num_classes)

        mask = box_scores > score_threshold  # shape=(batch_size, m, num_classes)
        max_boxes_tensor = tf.constant(max_boxes, tf.int32)

        predict_boxes = tf.TensorArray(tf.float32, size=batch_size)

        def loop_body(b, predict_boxes):
            boxes_b = tf.gather(boxes, b)  # shape=(m, boxes)
            mask_b = tf.gather(mask, b)  # shape=(m, num_classes)
            box_scores_b = tf.gather(box_scores, b)

            boxes_ = []
            scores_ = []
            classes_ = []
            for c in range(self.num_classes):
                class_boxes = tf.boolean_mask(boxes_b, mask_b[:, c])
                class_box_scores = tf.boolean_mask(box_scores_b[:, c], mask_b[:, c])
                nms_indies = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor,
                                                          iou_threshold)
                class_boxes = tf.gather(class_boxes, nms_indies)
                class_box_scores = tf.gather(class_box_scores, nms_indies)
                classes = tf.ones_like(class_box_scores, tf.float32) * c
                boxes_.append(class_boxes)
                scores_.append(class_box_scores)
                classes_.append(classes)
            boxes_ = tf.concat(boxes_, axis=0)  # shape=(?, 4)
            scores_ = tf.concat(scores_, axis=0)  # shape=(?, 1)
            classes_ = tf.concat(classes_, axis=0)  # shape=(?, 1)
            predict_boxes_b = tf.concat([boxes_, tf.reshape(scores_, [-1, 1]), tf.reshape(classes_, [-1, 1])], axis=-1)
            predict_boxes_b = tf.cond(
                tf.greater(tf.shape(boxes_)[0], max_boxes_tensor),
                lambda: tf.gather(predict_boxes_b, tf.nn.top_k(scores_, k=max_boxes).indices),
                lambda: tf.pad(predict_boxes_b, paddings=[[0, max_boxes_tensor - tf.shape(predict_boxes_b)[0]], [0, 0]],
                               mode='CONSTANT')
            )
            predict_boxes = predict_boxes.write(b, predict_boxes_b)
            return b + 1, predict_boxes

        _, predict_boxes = tf.while_loop(lambda b, predict_boxes: b < batch_size, loop_body, [0, predict_boxes])
        predict_boxes = predict_boxes.stack()
        predict_boxes = tf.reshape(predict_boxes, [batch_size, max_boxes_tensor, 6])
        ymin, xmin, ymax, xmax, confidence, class_id = tf.split(predict_boxes, num_or_size_splits=6, axis=-1)
        predict_boxes = tf.concat([xmin, ymin, xmax, ymax, confidence, class_id], axis=-1)
        return predict_boxes
