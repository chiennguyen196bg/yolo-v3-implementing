import numpy as np
import tensorflow as tf
from data_loader import DataReader
from yolov3 import config as cfg
import os
import time
from yolov3.model import Yolov3
from yolov3.util import load_weights
from metric import cal_AP
import argparse


def eval(test_dir, checkpoint, iou_threshold):
    input_shape_hw = np.array(cfg.INPUT_SHAPE, dtype=np.int32)[::-1]
    grid_shapes = [input_shape_hw // 32, input_shape_hw // 16, input_shape_hw // 8]

    test_tfrecord_files = [os.path.join(test_dir, x) for x in os.listdir(test_dir)]

    data_reader = DataReader(cfg.INPUT_SHAPE, cfg.ANCHORS, cfg.NUM_CLASSES, cfg.MAX_BOXES)
    test_dataset = data_reader.build_dataset(test_tfrecord_files, is_training=False, batch_size=cfg.TEST_BATCH_SIZE)
    # print(train_dataset.output_shapes)
    iterator = tf.data.Iterator. \
        from_structure(output_types=test_dataset.output_types,
                       output_shapes=(
                           tf.TensorShape([None, cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0], 3]),
                           tf.TensorShape([None, cfg.MAX_BOXES, 5]),
                           tf.TensorShape([None, grid_shapes[0][0], grid_shapes[0][1], 3, 5 + cfg.NUM_CLASSES]),
                           tf.TensorShape([None, grid_shapes[1][0], grid_shapes[1][1], 3, 5 + cfg.NUM_CLASSES]),
                           tf.TensorShape([None, grid_shapes[2][0], grid_shapes[2][1], 3, 5 + cfg.NUM_CLASSES])
                       ))
    test_init = iterator.make_initializer(test_dataset)
    test_init = iterator.make_initializer(test_dataset)

    images, bbox, bbox_true_13, bbox_true_26, bbox_true_52 = iterator.get_next()
    model = Yolov3(cfg.BATCH_NORM_DECAY, cfg.BATCH_NORM_EPSILON, cfg.LEAKY_RELU, cfg.ANCHORS, cfg.NUM_CLASSES)
    output = model.yolo_inference(images, False)

    saver = tf.train.Saver()

    predict_box = model.yolo_predict(output, image_shape=input_shape_hw, max_boxes=cfg.MAX_BOXES, score_threshold=0.3)

    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:

        saver.restore(sess, checkpoint)

        # Calculate mAP
        predict_ = []
        grouth_truth_ = []
        sess.run(test_init)
        try:
            while True:
                out_predict_box, out_bbox = sess.run([predict_box, bbox])
                predict_.append(out_predict_box)
                grouth_truth_.append(out_bbox)
        except tf.errors.OutOfRangeError:
            pass
        grouth_truth_ = np.concatenate(grouth_truth_, axis=0)
        predict_ = np.concatenate(predict_, axis=0)
        AP = cal_AP(predict_, grouth_truth_, cfg.NUM_CLASSES, iou_threshold)
        mAP = np.mean(AP)
        print('AP', AP, 'mAP:', mAP)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detecting for image')
    parser.add_argument('-ds', '--dataset', default='',
                        help='link to the image')
    parser.add_argument('-cp', '--checkpoint', default='./checkpoint',
                        help='checkpoint of model')
    parser.add_argument('-iou', '--threshold', type=float, default=0.5)
    args = parser.parse_args()
    eval(args.dataset, args.checkpoint, args.threshold)
