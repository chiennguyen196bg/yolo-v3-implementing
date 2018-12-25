import numpy as np
import tensorflow as tf
import cv2
from yolov3.model import Yolov3
from yolov3 import config as cfg
from yolov3.util import letterbox_image_opencv


def detect(video_path, class_names, model_path, out_video_path='out'):
    input_image = tf.placeholder(dtype=tf.float32, shape=[None, cfg.INPUT_SHAPE[0], cfg.INPUT_SHAPE[1], 3])
    input_image_shape = tf.placeholder(dtype=tf.int32, shape=(None, 2))
    model = Yolov3(cfg.BATCH_NORM_DECAY, cfg.BATCH_NORM_EPSILON, cfg.LEAKY_RELU, cfg.ANCHORS, len(class_names))
    yolo_outputs = model.yolo_inference(input_image, is_training=False)
    predict_boxes = model.yolo_predict(yolo_outputs, input_image_shape, score_threshold=.5)

    sess, cap, out = None, None, None
    out_video_path = out_video_path + '.mp4'
    try:
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        # print('Fps: ', fps)
        out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width, frame_height))
        while cap.isOpened():
            ret, image = cap.read()
            img_resized = letterbox_image_opencv(image, cfg.INPUT_SHAPE)
            image_data = img_resized / 255.
            image_data = np.expand_dims(image_data, axis=0)
            out_predict_boxes = sess.run(
                predict_boxes,
                feed_dict={
                    input_image: image_data,
                    input_image_shape: [image.shape[0:2]]
                }
            )

            out_predict_boxes = out_predict_boxes[0]
            out_predict_boxes = out_predict_boxes[out_predict_boxes[:, 4] > 0]

            for i, p_box in enumerate(out_predict_boxes):
                xmin, ymin, xmax, ymax, confidence, class_id = p_box
                class_name = class_names[int(class_id)]
                label = "{} {:.2f}%".format(class_name, confidence * 100.)
                image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=3)
                image = cv2.putText(image, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow('frame', image)
            out.write(image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        if cap:
            cap.release()
        if out:
            out.release()
        if sess:
            sess.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    detect('./data/IMG_6992.m4v', ['coconut'], './checkpoint/model.ckpt-30')
