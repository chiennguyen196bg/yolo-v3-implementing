import cv2
import numpy as np
from yolov3.detector import Detector
from tracking.tracker import Tracker
import tensorflow as tf


def track():
    class_names = ['coconut']
    cap, sess, out = None, None, None
    try:
        cap = cv2.VideoCapture('./data/IMG_6992.m4v')
        sess = tf.Session()

        detector = Detector('./checkpoint/model.ckpt-30', len(class_names), sess=sess)

        tracker = Tracker(0.3, 10, 10)

        rand_colors = np.random.randint(0, 255, (100, 3), dtype=np.uint)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter('./data/out-tracking.avi', cv2.VideoWriter_fourcc(*'XVID'), fps,
                              (frame_width, frame_height))

        while cap.isOpened():
            ret, image = cap.read()
            detection_boxes = detector.predict(image)
            tracker.update(detection_boxes)

            for trk in tracker.tracks:
                bbox = trk.get_state()
                # print(bbox)
                xmin, ymin, xmax, ymax = bbox.reshape((4,)).astype(int)
                color = rand_colors[trk.track_id % 100]
                color = (int(color[0]), int(color[1]), int(color[2]))
                image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=color, thickness=3)

            label = "{}: {}".format(class_names[0], tracker.track_id_count)
            # print(label)
            cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            # print(tracker.track_id_count)
            cv2.imshow('image', image)
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
    track()
