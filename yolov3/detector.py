import numpy as np
import tensorflow as tf
from yolov3.util import letterbox_image_opencv
from yolov3 import config as cfg
from yolov3.model import Yolov3


class Detector(object):
    def __init__(self, model_path, input_shape, num_classes, score_threshold=0.6, sess=None):
        self.model_path = model_path
        if sess is None:
            sess = tf.Session()
        self.sess = sess
        self.input_shape = input_shape
        self.input_image = tf.placeholder(dtype=tf.float32, shape=[None, input_shape[1], input_shape[0], 3])
        self.input_image_shape = tf.placeholder(dtype=tf.int32, shape=(None, 2))
        model = Yolov3(cfg.BATCH_NORM_DECAY, cfg.BATCH_NORM_EPSILON, cfg.LEAKY_RELU, cfg.ANCHORS, num_classes)
        yolo_outputs = model.yolo_inference(self.input_image, is_training=False)
        self.predict_boxes = model.yolo_predict(yolo_outputs, self.input_image_shape, score_threshold=score_threshold,
                                                iou_threshold=0.5)

        saver = tf.train.Saver()
        saver.restore(self.sess, model_path)

    def predict(self, image):
        image_shape = image.shape[0:2]
        resized_image = letterbox_image_opencv(image, self.input_shape)
        image_data = resized_image / 255.
        image_data = np.expand_dims(image_data, axis=0)
        predict_boxes = self.sess.run(self.predict_boxes,
                                      feed_dict={
                                          self.input_image: image_data,
                                          self.input_image_shape: [image_shape]
                                      })
        predict_boxes = predict_boxes[0]
        predict_boxes = predict_boxes[predict_boxes[:, 4] > 0]

        return predict_boxes


if __name__ == '__main__':
    import cv2

    class_names = ['coconut']
    image = cv2.imread('../data/14.png')

    detector = Detector('../checkpoint/model.ckpt-30', len(class_names))
    predict_boxes = detector.predict(image)

    for i, p_box in enumerate(predict_boxes):
        xmin, ymin, xmax, ymax, confidence, class_id = p_box
        class_name = class_names[int(class_id)]
        label = "{} {:.2f}%".format(class_name, confidence * 100.)
        image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=3)
        image = cv2.putText(image, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('frame', image)
    cv2.waitKey(5000)
