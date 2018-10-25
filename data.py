import os
import numpy as np
import tensorflow as tf


# def preprocess_true_boxes(true_boxs, input_shape, grid_shape, anchors, num_classes):
#     """
#     set true_boxs in to grid, convinient for calculating loss
#     :param true_boxs: list of true box has shape=(batch_size, max_objects, 5), x1y1x2y2class_id
#     :param input_shape: shape of input image: (ih, iw)
#     :param grip_shape: shape of output grid: (h, w)
#     :param anchors: anchors, shape=(n, 2)
#     :param num_classes: number of classes: integer
#     :return: a grip has shape [batch_size, h, w, (x + y + w + h + true_classes)]
#     """
#     assert (true_boxs[..., 4] < num_classes).all()
#     true_boxs = np.array(true_boxs, dtype="float32")  # shape=(batch_size, max_objects, 5)
#     input_shape = np.array(input_shape, dtype="int32")
#     grid_shape = np.array(grid_shape, dtype="int32")
#
#     # convert the numbers into scale of input shape
#     boxes_xy = (true_boxs[..., 0:2] + true_boxs[..., 2:4]) // 2
#     boxes_wh = true_boxs[..., 2:4] - true_boxs[..., 0:2]
#     true_boxs[..., 0:2] = boxes_xy / input_shape[::-1]
#     true_boxs[..., 2:4] = boxes_wh / input_shape[::-1]
#
#     batch_size = true_boxs.shape[0]
#     y_true = np.zeros((batch_size, grid_shape[0], grid_shape[1], len(anchors), 5 + num_classes), dtype='float32')
#
#     # Expand dim to apply broadcasting
#     anchors = np.expand_dims(anchors, 0)  # shape=(1, N, 2)
#     anchor_maxes = anchors / 2
#     anchor_mins = -anchor_maxes
#
#     valid_mask = boxes_wh[..., 0] > 0  # shape(m, T)
#
#     for b in batch_size:
#         # Discard zero rows
#         wh = boxes_wh[b, valid_mask[b]]  # shape=(t, 2)
#         if len(wh) == 0:
#             continue
#         # Expand dim to apply broadcasting
#         wh = np.expand_dims(wh, -2)  # shape=(t, 1, 2)
#         box_maxes = wh / 2
#         box_mins = -box_maxes
#
#         intersect_mins = np.maximum(box_mins, anchor_mins) # shape(t, N, 2)
#         intersect_maxes = np.minimum(box_maxes, anchor_maxes) # same shape
#         intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
#         interset_area = intersect_wh[..., 0] * intersect_wh[..., 1] # shape=(t, N)
#
#
#     return None

def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    """
    Preprocess true boxes for training input format
    :param true_boxes: array, shape(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape
    :param input_shape: array-like, hw, multiples of 32
    :param anchors: array, shape(N, 2), wh
    :param num_classes: integer
    :return:
        y_true: list of array, shape like yolo_output, xywh are reletive value
    """
    assert (true_boxes[..., 4] < num_classes).all()
    num_layers = len(anchors) // 3
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

    true_boxes = np.array(true_boxes, dtype="float32")
    input_shape = np.array(input_shape, dtype="int32")

    # convert true boxes to (m, T, 5) with x, y, w, h in scale of input shape
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    batch_size = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
    y_true = [
        np.zeros((batch_size, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask), 5 + num_classes), dtype='float32')
        for l in range(num_layers)]

    # Expand dim to apply broadcasting
    anchors = np.expand_dims(anchors, 0)  # shape=(1, N, 2)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0  # shape=(m, T)

    for b in range(batch_size):
        # Discard zero rows
        wh = boxes_wh[b, valid_mask[b]]  # shape=(t,2)
        if len(wh) == 0:
            continue
        # print(boxes_xy[b])
        # Expand dim to apply broadcasting
        wh = np.expand_dims(wh, -2)  # shape=(t, 1, 2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)  # shape=(t, N, 2)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]  # shape=(t,1)
        anchor_area = anchors[..., 0] * anchors[..., 1]  # shape=(1, N)
        iou = intersect_area / (box_area + anchor_area - intersect_area)  # shape=(t, N)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for true_box_i, anchor_i in enumerate(best_anchor):
            for layer_i in range(num_layers):
                if anchor_i in anchor_mask[layer_i]:
                    grid_x = np.floor(true_boxes[b, true_box_i, 0] * grid_shapes[layer_i][1]).astype('int32')
                    grid_y = np.floor(true_boxes[b, true_box_i, 1] * grid_shapes[layer_i][0]).astype('int32')
                    # print(grid_x, grid_y)
                    k = anchor_mask[layer_i].index(anchor_i)
                    c = true_boxes[b, true_box_i, 4].astype('int32')
                    # print(y_true[layer_i].shape)
                    # print(true_boxes[batch_size, true_box_i, 0:4])
                    # print(layer_i, b, grid_y, grid_x, k)
                    y_true[layer_i][b, grid_y, grid_x, k, 0:4] = true_boxes[b, true_box_i, 0:4]
                    y_true[layer_i][b, grid_y, grid_x, k, 4] = 1
                    y_true[layer_i][b, grid_y, grid_x, k, 5 + c] = 1

    return y_true


def _parse(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string),
            'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64)
        }
    )
    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)
    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, axis=0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, axis=0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, axis=0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, axis=0)
    label = tf.expand_dims(features['image/object/bbox/label'].values, axis=0)
    label = tf.cast(label, tf.float32)
    bbox = tf.concat([xmin, ymin, xmax, ymax, label], axis=0)
    bbox = tf.transpose(bbox, [1, 0])

    return image


def build_dataset(filenames, is_training, batch_size=32, buffer_size=2048):
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.map(_parse, num_parallel_calls=8)
    if is_training:
        dataset = dataset.shuffle(buffer_size=buffer_size).repeat(None)
    else:
        dataset = dataset.repeat(1)
    dataset = dataset.batch(batch_size).prefetch(batch_size)
    return dataset


if __name__ == '__main__':
    DATASET_DIR = "dataset/nfpa/"
    dataset = build_dataset(os.path.join(os.curdir, DATASET_DIR, 'train.tfrecords'), is_training=True, batch_size=6)
    iterator = dataset.make_one_shot_iterator()
    test = iterator.get_next()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            while True:
                test_ = sess.run([test])
                print(test_[0].shape)
        except tf.errors.OutOfRangeError:
            pass
