import tensorflow as tf
from model import *
import numpy as np

_ANCHORS = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]


def test_model():
    test_image = np.random.randint(0, 255, (416, 416, 3), dtype='int32')
    inputs = tf.placeholder('int32', (None, 416, 416, 3), 'inputs')
    raw_detect_1, raw_detect_2, raw_detect_3 = yolo_body(inputs, 20, True)
    grid, raw_detect, box_xy, box_wh = yolo_head(raw_detect_1, _ANCHORS[6:9], 20, tf.constant([416, 416]), True)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        raw_detect_1_, raw_detect_2_, raw_detect_3_ = sess.run([raw_detect_1, raw_detect_2, raw_detect_3],
                                                               {inputs: [test_image]})
        print(raw_detect_1_.shape)
        print(raw_detect_2_.shape)
        print(raw_detect_3_.shape)

        grid_, raw_detect_, box_xy_, box_wh_ = sess.run([grid, raw_detect, box_xy, box_wh], {inputs: [test_image]})

        print(grid_.shape)
        print(raw_detect_.shape)
        print(box_xy_.shape)
        print(box_wh_.shape)
        print(grid_[4, 6])


if __name__ == '__main__':
    test_model()
