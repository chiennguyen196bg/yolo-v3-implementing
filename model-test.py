import tensorflow as tf
from model import *
from data import *
import numpy as np
from loss import *

_ANCHORS = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]
anchors = np.array(_ANCHORS, dtype='float32')


def test_model():
    test_image = np.random.randint(0, 255, (416, 416, 3), dtype='int32')

    true_boxes = np.zeros((1, 3, 5))
    true_boxes[0, 0] = [163 - 78, 173 - 99, 163 + 78, 173 + 99, 5]
    true_boxes[0, 1] = [35 - 5, 49 - 7, 35 + 5, 49 + 7, 4]
    y_true = preprocess_true_boxes(true_boxes, (416, 416), anchors, 20)

    inputs = tf.placeholder('int32', (None, 416, 416, 3), 'inputs')
    raw_detect_1, raw_detect_2, raw_detect_3 = yolo_body(inputs, 20, True)
    # grid, raw_detect, box_xy, box_wh = yolo_head(raw_detect_1, _ANCHORS[6:9], 20, tf.constant([416, 416]), True)
    loss = yolo_loss([raw_detect_1, raw_detect_2, raw_detect_3], y_true, anchors, 20)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        raw_detect_1_, raw_detect_2_, raw_detect_3_, loss_ = sess.run([raw_detect_1, raw_detect_2, raw_detect_3, loss],
                                                               {inputs: [test_image]})
        print(raw_detect_1_.shape)
        print(raw_detect_2_.shape)
        print(raw_detect_3_.shape)
        print(loss_)

        # grid_, raw_detect_, box_xy_, box_wh_ = sess.run([grid, raw_detect, box_xy, box_wh], {inputs: [test_image]})
        #
        # print(grid_.shape)
        # print(raw_detect_.shape)
        # print(box_xy_.shape)
        # print(box_wh_.shape)
        # print(grid_[4, 6])



if __name__ == '__main__':
    test_model()
