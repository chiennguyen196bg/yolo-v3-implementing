from util import *
import numpy as np

_ANCHORS = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]


def test_preprocess_true_boxes():
    anchors = np.array(_ANCHORS, dtype='float32')
    input_shape = (416, 416)
    num_classes = 20
    batch_size = 5
    max_objects = 2
    true_boxes = np.zeros((batch_size, max_objects, 5))
    true_boxes[0, 0] = [163 - 78, 173 - 99, 163 + 78, 173 + 99, 5]
    true_boxes[1, 0] = [35 - 5, 49 - 7, 35 + 5, 49 + 7, 4]
    y_true = preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes)
    for y in y_true:
        print(y.shape)
    print(y_true[0][0, 5, 5, 1])
    print(y_true[2][1, 6, 4, 0])


if __name__ == '__main__':
    test_preprocess_true_boxes()
