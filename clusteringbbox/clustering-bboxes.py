import os
import tensorflow as tf
from yolov3 import config as cfg
import numpy as np

from clusteringbbox.kmeans import kmeans, avg_iou
from data_loader import DataReader

CLUSTERS = 9


def load_dataset():
    dataset_bboxes = []
    # for xml_file in glob.glob("{}/*xml".format(path)):
    # 	tree = ET.parse(xml_file)
    #
    # 	height = int(tree.findtext("./size/height"))
    # 	width = int(tree.findtext("./size/width"))
    #
    # 	for obj in tree.iter("object"):
    # 		xmin = int(obj.findtext("bndbox/xmin")) / width
    # 		ymin = int(obj.findtext("bndbox/ymin")) / height
    # 		xmax = int(obj.findtext("bndbox/xmax")) / width
    # 		ymax = int(obj.findtext("bndbox/ymax")) / height
    #
    # 		dataset.append([xmax - xmin, ymax - ymin])

    TRAINING_DATASET_DIR = "../dataset/pedestrian-dataset/train"
    VALIDATING_DATASET_DIR = "../dataset/pedestrian-dataset/val"

    train_tfrecord_files = [os.path.join(TRAINING_DATASET_DIR, x) for x in os.listdir(TRAINING_DATASET_DIR)]
    validating_tfrecord_files = [os.path.join(VALIDATING_DATASET_DIR, x) for x in os.listdir(VALIDATING_DATASET_DIR)]

    reader = DataReader(cfg.INPUT_SHAPE, cfg.ANCHORS, 1)
    dataset = reader.build_dataset(train_tfrecord_files + validating_tfrecord_files, is_training=False, batch_size=6)
    iterator = dataset.make_one_shot_iterator()
    image, bbox, bbox_true_13, bbox_true_26, bbox_true_52 = iterator.get_next()

    height = cfg.INPUT_SHAPE[0]
    width = cfg.INPUT_SHAPE[1]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            while True:
                bbox_out = sess.run(bbox)
                for boxes_in_an_image in bbox_out:
                    boxes_in_an_image = boxes_in_an_image[boxes_in_an_image[..., 3] > 0]
                    for b in boxes_in_an_image:
                        xmin, ymin, xmax, ymax = b[0:4]
                        if xmax - xmin == 0 or ymax - ymin == 0:
                            print(xmin, ymin, xmax, ymax)
                            continue
                        dataset_bboxes.append([xmax - xmin, ymax - ymin])
        except tf.errors.OutOfRangeError:
            pass

    return np.array(dataset_bboxes)


if __name__ == '__main__':
    data = load_dataset()
    out = kmeans(data, k=CLUSTERS)
    print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
    print("Boxes:\n {}".format(out))

    ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    print("Ratios:\n {}".format(sorted(ratios)))
