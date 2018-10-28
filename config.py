# Batch norm config
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-05
LEAKY_RELU = 0.1

ANCHORS = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]

INPUT_SHAPE = [416, 416]
NUM_CLASSES = 1

DATASET_DIR = "dataset/nfpa/"

LOG_DIR = './logs'

MAX_BOXES = 20
LEARNING_RATE = 0.001

PRE_TRAIN = True
MODEL_DIR = './test_model/model.ckpt-192192'
DARKNET53_WEIGHTS_PATH = './model-data/darknet53.weights'
N_EPOCHS = 50

TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
