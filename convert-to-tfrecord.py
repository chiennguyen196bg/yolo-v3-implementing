import os
import sys
import numpy as np
import cv2
import tensorflow as tf


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


def _convert_to_example(filename, image_bytes, label, bbox, height, width):
    _xcenter = []
    _ycenter = []
    _width = []
    _height = []
    for b in bbox:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([_xcenter, _ycenter, _width, _height], b)]
        # pylint: enable=expression-not-assigned

    label = label.astype(int).tolist()

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/object/bbox/xcenter': _float_feature(_xcenter),
        'image/object/bbox/ycenter': _float_feature(_ycenter),
        'image/object/bbox/width': _float_feature(_width),
        'image/object/bbox/height': _float_feature(_height),
        'image/object/bbox/label': _int64_feature(label),
        'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
        'image': _bytes_feature(image_bytes)
    }))
    return example


def convert(image_paths, out_path):
    print("Converting: ", out_path)
    num_images = len(image_paths)

    with tf.python_io.TFRecordWriter(out_path) as writer:
        for i, image_path in enumerate(image_paths):
            # _print_progress(i, num_images)
            image = cv2.imread(image_path)
            label_path = image_path.replace('jpg', 'txt')
            raw_label = np.loadtxt(label_path).reshape((-1,5))
            bbox = raw_label[..., 1:]
            label = raw_label[..., 0]
            if image_path.endswith("pos-294.jpg"):
                print("here")
                print(raw_label)
            example = _convert_to_example(image_path, image.tostring(), label, bbox, image.shape[0], image.shape[1])
            serialized = example.SerializeToString()
            writer.write(serialized)
    pass


if __name__ == '__main__':
    # print(os.path.basename("./dataset/nfpa/pos-174.txt"))
    # image = cv2.imread("./dataset/nfpa/pos-175.jpg")
    # print(image[0])
    #

    DATASET_PATH = 'dataset/nfpa/'
    with open(os.path.join(os.path.curdir, DATASET_PATH, 'train.txt')) as f:
        train_image_paths = f.readlines()
    train_image_paths = [x.strip() for x in train_image_paths]
    with open(os.path.join(os.path.curdir, DATASET_PATH, 'test.txt')) as f:
        test_image_paths = f.readlines()
    test_image_paths = [x.strip() for x in test_image_paths]
    convert(image_paths=test_image_paths, out_path=os.path.join(os.path.curdir, DATASET_PATH, 'test.tfrecords'))
    convert(image_paths=train_image_paths, out_path=os.path.join(os.path.curdir, DATASET_PATH, 'train.tfrecords'))


