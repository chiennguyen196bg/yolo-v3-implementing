import cv2
import numpy as np
from yolov3.detector import Detector
from tracking.tracker import Tracker
import tensorflow as tf
import time
import argparse


def track(input_path, input_shape, class_names, model_path, output):
    cap, sess, out = None, None, None
    try:
        cap = cv2.VideoCapture(input_path)
        sess = tf.Session()

        detector = Detector(model_path, input_shape, len(class_names), sess=sess)

        tracker = Tracker(0.3, 10, 10)

        rand_colors = np.random.randint(0, 255, (100, 3), dtype=np.uint)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'XVID'), fps,
                              (frame_width, frame_height))

        while cap.isOpened():
            ret, image = cap.read()
            start_time = time.time()
            detection_boxes = detector.predict(image)
            matched_trks = tracker.update(detection_boxes)

            for trk in matched_trks:
                bbox = trk.get_state()
                # print(bbox)
                xmin, ymin, xmax, ymax = bbox.reshape((4,)).astype(int)
                color = rand_colors[trk.track_id % 100]
                color = (int(color[0]), int(color[1]), int(color[2]))
                image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=color, thickness=2)
                image = cv2.putText(image, str(trk.track_id), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),
                                    2)

            # label = "{}: {}, fps: {}".format(class_names[0], tracker.num_track_is_tracked, int(1/(time.time() - start_time)))
            # print(label)
            # cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            # print(tracker.track_id_count)
            # cv2.imshow('image', image)
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
                        help='list class names: car,pedestrain,van')
    parser.add_argument('-cp', '--checkpoint', default='./checkpoint',
                        help='checkpoint of model')
    parser.add_argument('-o', '--output', default='./result.mp4',
                        help='output video')
    args = parser.parse_args()
    track(args.video_path, args.input_shape, args.class_names, args.checkpoint, args.output)
