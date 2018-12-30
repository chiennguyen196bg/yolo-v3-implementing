import numpy as np
import tensorflow as tf
from data_loader import DataReader
from yolov3 import config as cfg
import os
import time
from yolov3.model import Yolov3
from yolov3.util import load_weights
from metric import cal_AP


def train():
    print('input shape:', cfg.INPUT_SHAPE)
    print('learning rate:', cfg.LEARNING_RATE)
    print('dataset dir:', cfg.DATASET_DIR)
    print('max boxes:', cfg.MAX_BOXES)
    print('pre train:', cfg.PRE_TRAIN)
    print('weight decay:', cfg.WEIGHT_DECAY)
    print('anchors:', cfg.ANCHORS)

    # logs_dir = cfg.LOG_DIR + datetime.datetime.now().strftime("%YY%mm%dd%HH%MM%SS")
    logs_dir = cfg.LOG_DIR

    input_shape_hw = np.array(cfg.INPUT_SHAPE, dtype=np.int32)[::-1]
    grid_shapes = [input_shape_hw // 32, input_shape_hw // 16, input_shape_hw // 8]

    train_dir = os.path.join(cfg.DATASET_DIR, 'train')
    test_dir = os.path.join(cfg.DATASET_DIR, 'test')
    train_tfrecord_files = [os.path.join(train_dir, x) for x in os.listdir(train_dir)]
    test_tfrecord_files = [os.path.join(test_dir, x) for x in os.listdir(test_dir)]

    data_reader = DataReader(cfg.INPUT_SHAPE, cfg.ANCHORS, cfg.NUM_CLASSES, cfg.MAX_BOXES)
    train_dataset = data_reader.build_dataset(train_tfrecord_files, is_training=True, batch_size=cfg.TRAIN_BATCH_SIZE)
    test_dataset = data_reader.build_dataset(test_tfrecord_files, is_training=False, batch_size=cfg.TEST_BATCH_SIZE)
    # print(train_dataset.output_shapes)
    iterator = tf.data.Iterator. \
        from_structure(output_types=train_dataset.output_types,
                       output_shapes=(
                           tf.TensorShape([None, cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0], 3]),
                           tf.TensorShape([None, cfg.MAX_BOXES, 5]),
                           tf.TensorShape([None, grid_shapes[0][0], grid_shapes[0][1], 3, 5 + cfg.NUM_CLASSES]),
                           tf.TensorShape([None, grid_shapes[1][0], grid_shapes[1][1], 3, 5 + cfg.NUM_CLASSES]),
                           tf.TensorShape([None, grid_shapes[2][0], grid_shapes[2][1], 3, 5 + cfg.NUM_CLASSES])
                       ))
    train_init = iterator.make_initializer(train_dataset)
    test_init = iterator.make_initializer(test_dataset)
    is_training = tf.placeholder(tf.bool, shape=[])

    images, bbox, bbox_true_13, bbox_true_26, bbox_true_52 = iterator.get_next()
    bbox_true = [bbox_true_13, bbox_true_26, bbox_true_52]
    model = Yolov3(cfg.BATCH_NORM_DECAY, cfg.BATCH_NORM_EPSILON, cfg.LEAKY_RELU, cfg.ANCHORS, cfg.NUM_CLASSES)
    output = model.yolo_inference(images, is_training, weight_decay=cfg.WEIGHT_DECAY)
    loss = model.yolo_loss(output, bbox_true, ignore_thresh=0.5)
    l2_loss = tf.losses.get_regularization_loss()
    loss += l2_loss
    tf.summary.scalar('loss', loss)
    list_vars = list(tf.global_variables())

    global_step = tf.Variable(0, trainable=False)
    # lr = tf.train.exponential_decay(cfg.LEARNING_RATE, global_step, decay_steps=8000, decay_rate=0.8)
    # tf.summary.scalar('learning-rate', lr)
    lr = tf.constant(cfg.LEARNING_RATE, dtype=tf.float32)
    merged_summary = tf.summary.merge_all()
    optimizer = tf.train.AdamOptimizer(lr)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        if cfg.PRE_TRAIN:
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='yolo')
            train_op = optimizer.minimize(loss=loss, global_step=global_step, var_list=train_vars)
        else:
            train_op = optimizer.minimize(loss=loss, global_step=global_step)
    init_variables = tf.global_variables_initializer()
    saver = tf.train.Saver(var_list=list_vars)

    predict_box = model.yolo_predict(output, image_shape=input_shape_hw, max_boxes=cfg.MAX_BOXES, score_threshold=0.3)

    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        # sess.run(init_variables)
        # load model if have a checkpoint
        ckpt = tf.train.get_checkpoint_state(cfg.MODEL_DIR)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Restore model', ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            initialize_uninitialized(sess)
        else:
            print('Did not find a checkpoint. Initialize weights')
            sess.run(init_variables)
        # load pre train model
        if cfg.PRE_TRAIN is True:
            load_ops = load_weights(tf.global_variables(scope="darknet53"), cfg.DARKNET53_WEIGHTS_PATH)
            print('Load darknet 52 weights')
            sess.run(load_ops)
        train_writer = tf.summary.FileWriter(os.path.join(logs_dir, 'training'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(logs_dir, 'testing'), sess.graph)

        global_step_value = 0
        for epoch in range(cfg.N_EPOCHS):
            # Train phrase
            sess.run(train_init)
            train_losses = []
            try:
                while True:
                    # start_time = time.time()
                    train_loss, summary_value, global_step_value, _ = sess.run(
                        [loss, merged_summary, global_step, train_op], {is_training: True})
                    # duration = time.time() - start_time
                    # examples_per_sec = float(duration) / cfg.TRAIN_BATCH_SIZE
                    # format_str = 'Epoch {} step {},  train loss = {} ( {} examples/sec; {} ''sec/batch)'
                    # print(format_str.format(epoch, global_step_value, train_loss, examples_per_sec, duration))
                    train_writer.add_summary(summary_value, global_step=global_step_value)
                    train_losses.append(train_loss)
            except tf.errors.OutOfRangeError:
                train_writer.add_summary(
                    summary=tf.Summary(
                        value=[tf.Summary.Value(tag='loss_per_epoch', simple_value=np.mean(train_losses))]),
                    global_step=epoch
                )
                train_losses.clear()

            print('Epoch:', epoch)

            # Test phrase
            sess.run(test_init)
            test_losses = []
            try:
                while True:
                    test_loss = sess.run(loss, {is_training: False})
                    test_losses.append(test_loss)
            except tf.errors.OutOfRangeError:
                pass

            test_loss = np.mean(test_losses)
            # format_str = 'Epoch {} step {}, test loss = {}'
            # print(format_str.format(epoch, global_step_value, test_loss))
            print('Epoch', epoch, 'loss', test_loss)
            test_writer.add_summary(
                summary=tf.Summary(value=[tf.Summary.Value(tag='loss', simple_value=test_loss)]),
                global_step=global_step_value
            )

            # Calculate mAP
            if epoch % 5 == 0:
                print("Calculate mAP on test set")
                predict_ = []
                grouth_truth_ = []
                sess.run(test_init)
                try:
                    while True:
                        out_predict_box, out_bbox = sess.run([predict_box, bbox], {is_training: False})
                        predict_.append(out_predict_box)
                        grouth_truth_.append(out_bbox)
                except tf.errors.OutOfRangeError:
                    pass
                grouth_truth_ = np.concatenate(grouth_truth_, axis=0)
                predict_ = np.concatenate(predict_, axis=0)
                AP = cal_AP(predict_, grouth_truth_, cfg.NUM_CLASSES, 0.5)
                mAP = np.mean(AP)
                test_writer.add_summary(
                    summary=tf.Summary(value=[tf.Summary.Value(tag='mAP', simple_value=mAP)]),
                    global_step=epoch
                )
                print('AP', AP, 'mAP:', mAP)
                del grouth_truth_
                del predict_
                del AP
                del mAP

            if epoch % 5 == 0:
                checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, 'model.ckpt-{}'.format(epoch))
                saver.save(sess, checkpoint_path)


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    # print([str(i.name) for i in not_initialized_vars])  # only for testing
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


if __name__ == '__main__':
    train()
