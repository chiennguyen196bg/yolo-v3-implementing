import numpy as np
from tracking.kalman_filter import KalmanFilterWrapper
from util import box_iou_numpy
from scipy.optimize import linear_sum_assignment


class Track(object):
    """
    Track class for every object to be tracked
    """

    def __init__(self, detection, track_id=None, max_trace_length=10):
        """
        :param detection: predicted [xmin, ymin, xmax, ymax] of object to be tracked
        :param track_id: indentification of each track object
        """
        self.track_id = track_id
        self.KF = KalmanFilterWrapper(detection[..., 0:4])
        self.skipped_frames = 0
        self.trace = []
        self.max_trace_length = max_trace_length
        self.hit_streak = 0

    def update(self, detection=None):
        if detection is not None:
            self.skipped_frames = 0
            self.hit_streak += 1
            self.KF.correct(detection[..., 0:4])
        else:
            self.skipped_frames += 1
            self.hit_streak = 0
            # self.KF.update_state_without_measurement()

        if len(self.trace) > self.max_trace_length:
            self.trace.pop(0)
        self.trace.append(self.get_state())

    def predict(self):
        return self.KF.predict()

    def get_state(self):
        return self.KF.get_state()


def associate_detections_to_trackers(detections, predictions, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(predictions) == 0:
        return np.empty((0, 2), dtype=np.uint32), np.arange(len(detections)), np.empty((0, 1), dtype=int)

    iou_matrix = box_iou_numpy(detections[..., 0:4], predictions[..., 0:4])
    iou_matrix = iou_matrix.reshape((len(detections), len(predictions)))

    detect_indeces, predict_indeces = linear_sum_assignment(-iou_matrix)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in detect_indeces:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t, trk in enumerate(predictions):
        if t not in predict_indeces:
            unmatched_trackers.append(t)

    # filter out matched with low IoU
    matches = []
    for d, t in zip(detect_indeces, predict_indeces):
        if iou_matrix[d, t] < iou_threshold:
            unmatched_detections.append(d)
            unmatched_trackers.append(t)
        else:
            matches.append(np.array([d, t]).reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=np.uint32)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Tracker(object):
    def __init__(self, iou_thresh, max_frames_to_skip, max_trace_length):
        """
        Initialize variables used by Tracker class
        :param iou_thresh: track will be deleted and new track will be created if iou between detection and prediction
        is smaller than iou_thresh
        :param max_frames_to_skip: maximum allowed frames to be skipped for the track object undetected
        :param max_trace_length: trace path history length
        :param track_count_id: start indentification of each track object
        """
        self.iou_thresh = iou_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.tracks = []
        self.track_id_count = 0

    def update(self, detections):
        # Must have to call this to predict next state of each tracks
        predictions = np.array([trk.predict() for trk in self.tracks]).reshape(-1, 4)
        if len(detections) == 0:
            for trk in self.tracks:
                trk.update()
            return

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(detections, predictions,
                                                                                   self.iou_thresh)
        for d, t in matched:
            self.tracks[t].update(detections[d])

        for t in unmatched_trks:
            self.tracks[t].update()

        for d in unmatched_dets:
            trk = Track(detections[d, :])
            self.tracks.append(trk)
            # self.track_id_count += 1

        # if tracks are not detected for a long time, remove them
        for t, trk in reversed(list(enumerate(self.tracks))):
            if trk.skipped_frames > self.max_frames_to_skip:
                self.tracks.pop(t)

    def give_track_id(self, track: Track):
        track.track_id = self.track_id_count
        self.track_id_count += 1



# test
def test():
    import cv2
    import time

    image_width, image_height = 1000, 1000

    detect_1 = np.array([0, 0, 20, 25, 0], dtype=np.uint)
    detect_2 = np.array([900, 900, 930, 940, 0], dtype=np.uint)
    detect_history = []

    random_colors = np.random.randint(0, 255, (100, 3), dtype=np.uint)
    for _ in range(100):
        detect_1 = detect_1 + 5
        detect_2 = detect_2 - 6
        noise_detect_1 = detect_1 + np.random.randint(90, 100, (5,), dtype=np.uint)
        noise_detect_2 = detect_2 + np.random.randint(50, 60, (5,), dtype=np.uint)
        detect_history.append(np.concatenate([noise_detect_1, noise_detect_2], axis=0).reshape(-1, 5))

    tracker = Tracker(0.3, 3, 10, 0)

    for det in detect_history:
        image = np.zeros((image_height, image_width, 3), dtype=np.int8)
        image = cv2.rectangle(image, (det[0][0], det[0][1]), (det[0][2], det[0][3]), color=(0, 255, 0), thickness=3)
        image = cv2.rectangle(image, (det[1][0], det[1][1]), (det[1][2], det[1][3]), color=(0, 255, 0), thickness=3)
        if np.random.rand() < 0.2:
            tracker.update(np.empty((0, 5), dtype=np.int8))
        else:
            tracker.update(det.reshape((-1, 5)))
        print(len(tracker.tracks))
        for trk in tracker.tracks:
            bbox = trk.get_state().reshape((-1)).astype(np.uint)
            color = random_colors[trk.track_id]
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                          color=(int(color[0]), int(color[1]), int(color[2])),
                          thickness=3)
        cv2.imshow('image', image)

        time.sleep(0.2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    test()
