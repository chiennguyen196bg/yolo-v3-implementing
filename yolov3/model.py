import numpy as np
import tensorflow as tf

# import config as cf

slim = tf.contrib.slim


def darknet53(inputs):
    """
    Builds Darknet-53 model.
    """
    inputs = _conv2d_fixed_padding(inputs, 32, 3)
    inputs = _conv2d_fixed_padding(inputs, 64, 3, stride=2)
    inputs = _darknet53_block(inputs, 32)
    inputs = _conv2d_fixed_padding(inputs, 128, 3, stride=2)

    for i in range(2):
        inputs = _darknet53_block(inputs, 64)

    inputs = _conv2d_fixed_padding(inputs, 256, 3, stride=2)

    for i in range(8):
        inputs = _darknet53_block(inputs, 128)

    route_1 = inputs
    inputs = _conv2d_fixed_padding(inputs, 512, 3, stride=2)

    for i in range(8):
        inputs = _darknet53_block(inputs, 256)

    route_2 = inputs
    inputs = _conv2d_fixed_padding(inputs, 1024, 3, stride=2)

    for i in range(4):
        inputs = _darknet53_block(inputs, 512)

    return route_1, route_2, inputs


def _conv2d_fixed_padding(inputs, filters, kernel_size, stride=1):
    if stride > 1:
        inputs = _fixed_padding(inputs, kernel_size)
    inputs = slim.conv2d(inputs, filters, kernel_size, stride=stride, padding=('SAME' if stride == 1 else 'VALID'))
    return inputs


def _darknet53_block(inputs, filters):
    shortcut = inputs
    inputs = _conv2d_fixed_padding(inputs, filters, 1)
    inputs = _conv2d_fixed_padding(inputs, filters * 2, 3)

    inputs = inputs + shortcut
    return inputs


@tf.contrib.framework.add_arg_scope
def _fixed_padding(inputs, kernel_size, *args, mode='CONSTANT', **kwargs):
    """
    Pads the input along the spatial dimensions independently of input size.
    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in].
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.
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

    new_height = out_shape[2]
    new_width = out_shape[1]

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


def yolo_boxes_and_scores():
    pass

def yolo_correct_boxes():
    pass


class Yolov3:
    def __init__(self, batch_norm_decay, batch_norm_epsilon, leaky_relu, anchors, num_classes):
        self.batch_norm_decay = batch_norm_decay
        self.batch_norm_epsilon = batch_norm_epsilon
        self.leaky_relu = leaky_relu
        self.anchors = np.array(anchors, dtype=np.float32).reshape(-1, 2)
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.anchors_tensor = tf.reshape(tf.constant(anchors), shape=(-1, 2))

    def yolo_inference(self, inputs, is_training=False, reuse=False):
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
                                biases_initializer=None,
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

            grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l], self.anchors[anchor_mask[l]], self.num_classes,
                                                         input_shape,
                                                         calc_loss=True)
            pred_box = tf.concat([pred_xy, pred_wh], axis=-1)  # shape=(m, grid_h, grid_w, num_anchors, 4)

            # Darknet raw box to calculate loss
            raw_true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid
            raw_true_wh = tf.log(y_true[l][..., 2:4] / self.anchors[anchor_mask[l]] * input_shape[::-1])
            raw_true_wh = tf.where(object_mask_concat, raw_true_wh, tf.zeros_like(raw_true_wh))
            box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

            # Find ignore mask, iterate over each of batch
            ignore_mask = tf.TensorArray(tf.float32, size=1, dynamic_size=True)

            def loop_body(b, ignore_mask):
                _y_true = tf.gather(y_true[l], b)
                _object_mask_bool = tf.gather(object_mask_bool, b)
                _pred_box = tf.gather(pred_box, b)
                true_box = tf.boolean_mask(_y_true[..., 0:4], _object_mask_bool[..., 0])  # shape=(t,4)
                iou = box_iou(_pred_box, true_box)  # shape=(grid_h, grid_w, num_anchor, num_true_box)
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

    def yolo_eval(self, yolo_outputs, image_shape, max_boxes=20, score_threshold=.6, iou_threshold=.5):
        """Evaluate YOLO model on given input and return filtered boxes."""
        num_layers = len(yolo_outputs)
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
        input_shape = tf.shape(yolo_outputs[0])[1:3] * 32



#
# def detections_boxes(detections):
#     """
#     Converts center x, center y, width and height values to coordinates of top left and bottom right points.
#     :param detections: outputs of YOLO v3 detector of shape (?, 10647, (num_classes + 5))
#     :return: converted detections of same shape as input
#     """
#     center_x, center_y, width, height, attrs = tf.split(detections, [1, 1, 1, 1, -1], axis=-1)
#     w2 = width / 2
#     h2 = height / 2
#     x0 = center_x - w2
#     y0 = center_y - h2
#     x1 = center_x + w2
#     y1 = center_y + h2
#
#     boxes = tf.concat([x0, y0, x1, y1], axis=-1)
#     detections = tf.concat([boxes, attrs], axis=-1)
#     return detections
#
#
# def _iou(box1, box2):
#     """
#     Computes Intersection over Union value for 2 bounding boxes
#
#     :param box1: array of 4 values (top left and bottom right coords): [x0, y0, x1, x2]
#     :param box2: same as box1
#     :return: IoU
#     """
#     b1_x0, b1_y0, b1_x1, b1_y1 = box1
#     b2_x0, b2_y0, b2_x1, b2_y1 = box2
#
#     int_x0 = max(b1_x0, b2_x0)
#     int_y0 = max(b1_y0, b2_y0)
#     int_x1 = min(b1_x1, b2_x1)
#     int_y1 = min(b1_y1, b2_y1)
#
#     int_area = (int_x1 - int_x0) * (int_y1 - int_y0)
#
#     b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
#     b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)
#
#     # we add small epsilon of 1e-05 to avoid division by 0
#     iou = int_area / (b1_area + b2_area - int_area + 1e-05)
#     return iou
#
#
# def non_max_suppression(predictions_with_boxes, confidence_threshold, iou_threshold=0.4):
#     """
#     Applies Non-max suppression to prediction boxes.
#     :param predictions_with_boxes: 3D numpy array, first 4 values in 3rd dimension are bbox attrs, 5th is confidence
#     :param confidence_threshold: the threshold for deciding if prediction is valid
#     :param iou_threshold: the threshold for deciding if two boxes overlap
#     :return: dict: class -> [(box, score)]
#     """
#     conf_mask = np.expand_dims((predictions_with_boxes[:, :, 4] > confidence_threshold), -1)
#     predictions = predictions_with_boxes * conf_mask
#
#     result = {}
#     for i, image_pred in enumerate(predictions):
#         shape = image_pred.shape
#         non_zero_idxs = np.nonzero(image_pred)
#         image_pred = image_pred[non_zero_idxs]
#         image_pred = image_pred.reshape(-1, shape[-1])
#
#         bbox_attrs = image_pred[:, :5]
#         classes = image_pred[:, 5:]
#         classes = np.argmax(classes, axis=-1)
#
#         unique_classes = list(set(classes.reshape(-1)))
#
#         for cls in unique_classes:
#             cls_mask = classes == cls
#             cls_boxes = bbox_attrs[np.nonzero(cls_mask)]
#             cls_boxes = cls_boxes[cls_boxes[:, -1].argsort()[::-1]]
#             cls_scores = cls_boxes[:, -1]
#             cls_boxes = cls_boxes[:, :-1]
#
#             while len(cls_boxes) > 0:
#                 box = cls_boxes[0]
#                 score = cls_scores[0]
#                 if not cls in result:
#                     result[cls] = []
#                 result[cls].append((box, score))
#                 cls_boxes = cls_boxes[1:]
#                 ious = np.array([_iou(box, x) for x in cls_boxes])
#                 iou_mask = ious < iou_threshold
#                 cls_boxes = cls_boxes[np.nonzero(iou_mask)]
#                 cls_scores = cls_scores[np.nonzero(iou_mask)]
#
#     return result
