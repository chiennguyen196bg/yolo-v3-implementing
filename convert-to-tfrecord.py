import os
import sys
import numpy as np
import cv2
import tensorflow as tf


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


def _convert_to_example(filename, image_buffer, label, bbox, height, width):
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bbox:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([xmin, ymin, xmax, ymax], b)]
        # pylint: enable=expression-not-assigned

    label = label.astype(int).tolist()

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/object/bbox/xmin': _float_feature(xmin),
        'image/object/bbox/ymin': _float_feature(ymin),
        'image/object/bbox/xmax': _float_feature(xmax),
        'image/object/bbox/ymax': _float_feature(ymax),
        'image/object/bbox/label': _int64_feature(label),
        'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
        'image/encoded': _bytes_feature(image_buffer)
    }))
    return example


def _process_image(filename, coder):
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()
        # Clean the dirty data.
    # if _is_png(filename):
    #     # 1 image is a PNG.
    #     print('Converting PNG to JPEG for %s' % filename)
    #     image_data = coder.png_to_jpeg(image_data)
    # elif _is_cmyk(filename):
    #     # 22 JPEG images are in CMYK colorspace.
    #     print('Converting CMYK to RGB for %s' % filename)
    #     image_data = coder.cmyk_to_rgb(image_data)

        # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3
    return image_data, height, width


def _read_annotation(annotation_path):
    annotation = np.loadtxt(annotation_path).reshape((-1, 5))
    xy = annotation[..., 1:3]
    wh = annotation[..., 3:5]
    wh_half = wh / 2
    xy_min = xy - wh_half
    xy_max = xy + wh_half
    bbox = np.concatenate([xy_min, xy_max], axis=-1)
    label = annotation[..., 0]
    return bbox, label


def convert(image_paths, out_path):
    print("Converting: ", out_path)
    num_images = len(image_paths)
    coder = ImageCoder()
    with tf.python_io.TFRecordWriter(out_path) as writer:
        for i, image_path in enumerate(image_paths):
            _print_progress(i, num_images)
            image_data, height, width = _process_image(image_path, coder)
            annotation_path = image_path.replace('jpg', 'txt')
            bbox, label = _read_annotation(annotation_path)
            bbox[..., [0, 2]] = bbox[..., [0, 2]] * width
            bbox[..., [1, 3]] = bbox[..., [1, 3]] * height
            # print(bbox)
            example = _convert_to_example(image_path, image_data, label, bbox, height, width)
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
