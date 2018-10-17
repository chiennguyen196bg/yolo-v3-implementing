import numpy as np
import tensorflow as tf


def preprocess_true_boxes(true_boxs, input_shape, grid_shape, num_classes):
    """
    set true_boxs in to grid, convinient for calculating loss
    :param true_boxs: list of true box has shape: [batch_size, k, (class_id: x: y: w: h)] (k is the max object per
    a image in whole data)
    :param input_shape: shape of input image: (ih, iw)
    :param grip_shape: shape of output grid: (h, w)
    :param num_classes: number of classes: integer
    :return: a grip has shape [batch_size, h, w, (x + y + w + h + true_classes)]
    """
    batch_size = true_boxs.shape[0]
    
    return None
