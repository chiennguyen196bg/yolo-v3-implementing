import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from random import shuffle
import math


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that converts CMYK JPEG data to RGB JPEG data.
        self._cmyk_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
        self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def cmyk_to_rgb(self, image_data):
        return self._sess.run(self._cmyk_to_rgb,
                              feed_dict={self._cmyk_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _print_progress(count, total):
    # Percentage completion.
    pct_complete = float(count) / total

    # Status-message.
    # Note the \r which means the line should overwrite itself.
    msg = "\r- Progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, bbox):
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bbox:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([xmin, ymin, xmax, ymax], b)]
        # pylint: enable=expression-not-assigned

    label = [int(x) for x in label]

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/object/bbox/xmin': _float_feature(xmin),
        'image/object/bbox/ymin': _float_feature(ymin),
        'image/object/bbox/xmax': _float_feature(xmax),
        'image/object/bbox/ymax': _float_feature(ymax),
        'image/object/bbox/label': _int64_feature(label),
        'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
        'image/encoded': _bytes_feature(image_buffer)
    }))
    return example


def _process_image_path(coder, filename):
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()

    if filename.endswith('.png'):
        # print('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)

    image = coder.decode_jpeg(image_data)

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width


# def _read_annotation(annotation_path):
#     annotation = np.loadtxt(annotation_path).reshape((-1, 5))
#     xy = annotation[..., 1:3]
#     wh = annotation[..., 3:5]
#     wh_half = wh / 2
#     xy_min = xy - wh_half
#     xy_max = xy + wh_half
#     bbox = np.concatenate([xy_min, xy_max], axis=-1)
#     label = annotation[..., 0]
#     return bbox, label
#
#
# def convert(image_paths, out_path):
#     print("Converting: ", out_path)
#     num_images = len(image_paths)
#     with tf.python_io.TFRecordWriter(out_path) as writer:
#         for i, image_path in enumerate(image_paths):
#             _print_progress(i, num_images)
#             image_data = _process_image(image_path)
#             annotation_path = image_path.replace('jpg', 'txt')
#             bbox, label = _read_annotation(annotation_path)
#             example = _convert_to_example(image_path, image_data, label, bbox)
#             serialized = example.SerializeToString()
#             writer.write(serialized)
#     pass


def _extract_data_from_annotations(_annotations):
    image_paths = []
    bboxes = []
    labels = []
    for annotation in _annotations:
        image_path = annotation[0]
        bbox_in_an_image = []
        labels_in_an_image = []
        for raw_bbox in annotation[1:]:
            bbox_with_label = raw_bbox.split(',')
            bbox_with_label = [float(x) for x in bbox_with_label]
            bbox_in_an_image.append(bbox_with_label[0:4])
            labels_in_an_image.append(bbox_with_label[4])
        image_paths.append(image_path)
        bboxes.append(bbox_in_an_image)
        labels.append(labels_in_an_image)
    return image_paths, bboxes, labels


def _process_annotations(name, coder, image_paths, bboxes, labels, num_shards=1):
    assert len(image_paths) == len(bboxes)
    assert len(image_paths) == len(labels)
    shard_ranges = np.linspace(0, len(image_paths), num_shards + 1)
    print(shard_ranges)

    processed_file_counter = 0

    for shard in range(num_shards):
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(os.curdir, name, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        files_in_shard = np.arange(shard_ranges[shard], shard_ranges[shard + 1], dtype=int)
        print(files_in_shard)
        for i in files_in_shard:
            # print(image_paths[i])
            image_buffer, height, width = _process_image_path(coder, image_paths[i])
            bboxes_np = np.array(bboxes[i])
            bboxes_np = bboxes_np / np.array([width, height, width, height])
            example = _convert_to_example(image_paths[i], image_buffer, labels[i], bboxes_np)
            serialized = example.SerializeToString()
            writer.write(serialized)
            processed_file_counter += 1
        writer.close()

    print(processed_file_counter)


def main():
    coder = ImageCoder()

    with open('./annotations.txt') as f:
        annotations = f.readlines()

    annotations = [x.strip() for x in annotations]
    annotations = [x.split(' ') for x in annotations]
    shuffle(annotations)

    VAL_RATIO = 0.3
    n_val = math.floor(VAL_RATIO * len(annotations))
    val_annotations = annotations[:n_val]
    train_annotations = annotations[n_val:]

    train_image_paths, train_bboxes, train_labels = _extract_data_from_annotations(train_annotations)
    val_image_paths, val_bboxes, val_labels = _extract_data_from_annotations(val_annotations)
    print(len(train_image_paths), len(val_image_paths))

    _process_annotations('train', coder, train_image_paths, train_bboxes, train_labels, 10)
    _process_annotations('val', coder, val_image_paths, val_bboxes, val_labels, 3)


if __name__ == '__main__':
    # print(os.path.basename("./dataset/nfpa/pos-174.txt"))
    # image = cv2.imread("./dataset/nfpa/pos-175.jpg")
    # print(image[0])
    #

    # DATASET_PATH = 'dataset/nfpa/'
    # with open(os.path.join(os.path.curdir, DATASET_PATH, 'train.txt')) as f:
    #     train_image_paths = f.readlines()
    # train_image_paths = [x.strip() for x in train_image_paths]
    # with open(os.path.join(os.path.curdir, DATASET_PATH, 'test.txt')) as f:
    #     test_image_paths = f.readlines()
    # test_image_paths = [x.strip() for x in test_image_paths]
    # convert(image_paths=test_image_paths, out_path=os.path.join(os.path.curdir, DATASET_PATH, 'test.tfrecords'))
    # convert(image_paths=train_image_paths, out_path=os.path.join(os.path.curdir, DATASET_PATH, 'train.tfrecords'))

    main()
