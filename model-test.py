import tensorflow as tf
from model import *
import numpy as np


def test_model():
    test_image = np.random.randint(0, 255, (416, 416, 3), dtype='int32')
    inputs = tf.placeholder('int32', (None, 416, 416, 3), 'inputs')
    raw_detect_1, raw_detect_2, raw_detect_3 = yolo_body(inputs, 20, True)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        raw_detect_1_, raw_detect_2_, raw_detect_3_ = sess.run([raw_detect_1, raw_detect_2, raw_detect_3],
                                                               {inputs: [test_image]})
        print(raw_detect_1_.shape)
        print(raw_detect_2_.shape)
        print(raw_detect_3_.shape)


if __name__ == '__main__':
    test_model()
