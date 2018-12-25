import os
import numpy as np
import tensorflow as tf
from yolov3 import config as cfg
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class DataReader:
    def __init__(self, input_shape, anchors, num_classes, max_boxes=20):
        self.input_shape = np.array(input_shape, np.int32)
        self.anchors = np.array(anchors, np.float32).reshape(-1, 2)
        self.num_classes = num_classes
        self.input_shape_tensor = tf.constant(self.input_shape)
        self.anchors_tensor = tf.constant(self.anchors)
        self.max_boxes = max_boxes

    def _preprocess_true_boxes(self, true_boxes):
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
        assert (true_boxes[..., 4] < self.num_classes).all()
        num_layers = len(self.anchors) // 3
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        true_boxes = np.array(true_boxes, dtype='float32')
        input_shape = self.input_shape
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2.
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
        true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
        true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

        grid_shapes = [input_shape // 32, input_shape // 16, input_shape // 8]
        y_true = [
            np.zeros((grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + self.num_classes), dtype='float32')
            for l in range(num_layers)]

        anchors = np.expand_dims(self.anchors, 0)
        anchors_max = anchors / 2.
        anchors_min = -anchors_max

        valid_mask = boxes_wh[..., 0] > 0
        wh = boxes_wh[valid_mask]

        wh = np.expand_dims(wh, -2)
        boxes_max = wh / 2.
        boxes_min = -boxes_max

        intersect_min = np.maximum(boxes_min, anchors_min)
        intersect_max = np.minimum(boxes_max, anchors_max)
        intersect_wh = np.maximum(intersect_max - intersect_min, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        best_anchor = np.argmax(iou, axis=-1)
        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[t, 1] * grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[t, 4].astype('int32')
                    y_true[l][j, i, k, 0:4] = true_boxes[t, 0:4]
                    y_true[l][j, i, k, 4] = 1.
                    y_true[l][j, i, k, 5 + c] = 1.
        return y_true[0], y_true[1], y_true[2]

    def _preprocess(self, image, bbox):
        """resize image to input shape"""
        image_width, image_height = tf.cast(tf.shape(image)[1], tf.float32), tf.cast(tf.shape(image)[0], tf.float32)
        input_shape = tf.cast(self.input_shape_tensor, tf.float32)
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
                                                         tf.cast(input_height, tf.int32),
                                                         tf.cast(input_width, tf.int32))
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
        bbox = tf.clip_by_value(bbox, clip_value_min=0, clip_value_max=input_width - 1)
        bbox = tf.cond(tf.greater(tf.shape(bbox)[0], self.max_boxes), lambda: bbox[:self.max_boxes],
                       lambda: tf.pad(bbox, paddings=[[0, self.max_boxes - tf.shape(bbox)[0]], [0, 0]],
                                      mode='CONSTANT'))
        return image, bbox

    def _parse(self, serialized_example):
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
        image, bbox = self._preprocess(image, bbox)
        bbox_true_13, bbox_true_26, bbox_true_52 = tf.py_func(self._preprocess_true_boxes, [bbox],
                                                              [tf.float32, tf.float32, tf.float32])
        return image, bbox, bbox_true_13, bbox_true_26, bbox_true_52

    def build_dataset(self, filenames, is_training=False, batch_size=32, buffer_size=2048):
        dataset = tf.data.TFRecordDataset(filenames=filenames)
        dataset = dataset.map(self._parse, num_parallel_calls=8)
        if is_training:
            dataset = dataset.shuffle(buffer_size=buffer_size)

        dataset = dataset.batch(batch_size).prefetch(batch_size)
        return dataset


if __name__ == '__main__':
    DATASET_DIR = "dataset/pedestrian-dataset/train"
    tfrecord_files = [os.path.join(DATASET_DIR, x) for x in os.listdir(DATASET_DIR)]
    reader = DataReader(cfg.INPUT_SHAPE, cfg.ANCHORS, 1)
    dataset = reader.build_dataset(tfrecord_files, is_training=True, batch_size=6)
    iterator = dataset.make_one_shot_iterator()
    image, bbox, bbox_true_13, bbox_true_26, bbox_true_52 = iterator.get_next()
    fig, ax = plt.subplots(1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            while True:
                images_out, bbox_out, bbox_true_13_out, bbox_true_26_out, bbox_true_52_out = sess.run(
                    [image, bbox, bbox_true_13, bbox_true_26, bbox_true_52])
                m = 2
                sample_image = images_out[m]
                ax.imshow(sample_image)
                sample_bboxes = bbox_out[m]
                print(sample_bboxes)
                sample_bboxes = sample_bboxes[sample_bboxes[..., 3] > 0]

                for bbox in sample_bboxes:
                    xmin, ymin, xmax, ymax = bbox[0:4]

                    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r',
                                             facecolor='none')
                    ax.add_patch(rect)
                plt.show()

                break
        except tf.errors.OutOfRangeError:
            pass
