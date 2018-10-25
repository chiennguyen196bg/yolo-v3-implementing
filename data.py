import os
import numpy as np
import tensorflow as tf
import config as cfg
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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


def _preprocess(image, bbox, input_shape):
    """resize image to input shape"""
    image_width, image_height = tf.cast(tf.shape(image)[1], tf.float32), tf.cast(tf.shape(image)[0], tf.float32)
    input_shape = tf.cast(input_shape, tf.float32)
    input_height, input_width = input_shape[0], input_shape[1]
    scale = tf.minimum(input_width / image_width, input_height / image_height)
    new_height = image_height * scale
    new_width = image_width * scale
    dy = (input_height - new_height) / 2
    dx = (input_width - new_width) / 2
    # image = tf.Print(image, [image[125, 125]])
    image = tf.image.resize_images(image, [tf.cast(new_height, tf.int32), tf.cast(new_width, tf.int32)],
                                   method=tf.image.ResizeMethod.BICUBIC)
    new_image = tf.image.pad_to_bounding_box(image, tf.cast(dy, tf.int32), tf.cast(dx, tf.int32),
                                             tf.cast(input_height, tf.int32), tf.cast(input_width, tf.int32))
    image_ones = tf.ones_like(image)
    image_ones_padded = tf.image.pad_to_bounding_box(image_ones, tf.cast(dy, tf.int32), tf.cast(dx, tf.int32),
                                                     tf.cast(input_height, tf.int32), tf.cast(input_width, tf.int32))
    image_color_padded = (1 - image_ones_padded) * 128
    image = image_color_padded + new_image
    image = image / 255.
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    # bbox
    xmin, ymin, xmax, ymax, label = tf.split(value=bbox, num_or_size_splits=5, axis=1)
    xmin = xmin * new_width + dx
    xmax = xmax * new_width + dx
    ymin = ymin * new_height + dy
    ymax = ymax * new_height + dy
    bbox = tf.concat([xmin, ymin, xmax, ymax, label], 1)
    return image, bbox


def _parse(serialized_example, input_shape):
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
    bbox = tf.cond(tf.equal(tf.size(bbox), 0), lambda: tf.constant([[0] * 5], dtype=tf.float32), lambda: bbox)
    image, bbox = _preprocess(image, bbox, input_shape)
    return image, bbox[0]


def build_dataset(filenames, input_shape=cfg.INPUT_SHAPE, is_training=False, batch_size=32, buffer_size=2048):
    input_shape = tf.constant(np.array(input_shape))
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.map(lambda x: _parse(x, input_shape), num_parallel_calls=8)
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
    images, bbox = iterator.get_next()
    fig, ax = plt.subplots(1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            while True:
                images_out, bbox_out = sess.run([images, bbox])
                print(images_out[0][125][125])
                ax.imshow(images_out[2])
                xmin, ymin, xmax, ymax = bbox_out[2][0:4]
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r',
                                         facecolor='none')
                ax.add_patch(rect)
                plt.show()
                break
        except tf.errors.OutOfRangeError:
            pass
