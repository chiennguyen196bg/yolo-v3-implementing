import numpy as np
import tensorflow as tf
import cv2
from yolov3.model import Yolov3
from yolov3 import config as cfg
from yolov3.util import letterbox_image_opencv
import argparse


def detect(video_path, input_shape, class_names, model_path, out_video_path='out.mp4'):
    input_image = tf.placeholder(dtype=tf.float32, shape=[None, input_shape[1], input_shape[0], 3])
    input_image_shape = tf.placeholder(dtype=tf.int32, shape=(None, 2))
    model = Yolov3(cfg.BATCH_NORM_DECAY, cfg.BATCH_NORM_EPSILON, cfg.LEAKY_RELU, cfg.ANCHORS, len(class_names))
    yolo_outputs = model.yolo_inference(input_image, is_training=False)
    predict_boxes = model.yolo_predict(yolo_outputs, input_image_shape, score_threshold=.5)

    sess, cap, out = None, None, None
    # out_video_path = out_video_path + '.mp4'
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
                image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=1)
                image = cv2.putText(image, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

            # cv2.imshow('frame', image)
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
    parser = argparse.ArgumentParser(description='Detecting for image')
    parser.add_argument('-i', '--video-path', default='',
                        help='link to the image')
    parser.add_argument('-s', '--input-shape', nargs=2, type=int)
    parser.add_argument('-cn', '--class-names', nargs='+', type=str,
                        help='list class names, car,pedestrain,van')
    parser.add_argument('-cp', '--checkpoint', default='./checkpoint',
                        help='checkpoint of model')
    parser.add_argument('-o', '--output', default='./result.mp4',
                        help='output video')
    args = parser.parse_args()
    detect(args.video_path, args.input_shape, args.class_names, args.checkpoint, args.output)
