# Batch norm config
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-05
LEAKY_RELU = 0.1

# ANCHORS = [(5, 13), (8, 23), (20, 17), (12, 38), (33, 21), (19, 54), (41, 29), (36, 98), (72, 194)]
ANCHORS = [(4, 11), (5, 14), (10, 11), (7, 18), (8, 24), (11, 28), (12, 37), (19, 54), (44, 131)]

INPUT_SHAPE = [416, 416]
NUM_CLASSES = 1

DATASET_DIR = "dataset/MOT17Det/train"

LOG_DIR = './logs'

CHECKPOINT_DIR = './checkpoint'

MAX_BOXES = 52
LEARNING_RATE = 0.001

PRE_TRAIN = True
MODEL_DIR = './checkpoint'
DARKNET53_WEIGHTS_PATH = './model-data/darknet53.weights'
N_EPOCHS = 500

TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8

WEIGHT_DECAY = 0.0
