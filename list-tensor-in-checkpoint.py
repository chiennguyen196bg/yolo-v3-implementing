import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from yolov3 import config as cfg

if __name__ == '__main__':
    latest_ckp = tf.train.latest_checkpoint(cfg.MODEL_DIR)
    print_tensors_in_checkpoint_file(latest_ckp, all_tensors=False, tensor_name='')