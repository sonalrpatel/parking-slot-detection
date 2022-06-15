# ================================================================
#   File name   : configs.py
#   Author      : sonalrpatel
#   Created date: 27-12-2021
#   GitHub      : https://github.com/sonalrpatel/object-detection-yolo
#   Description : yolov3 configuration file
# ================================================================

# YOLO options
YOLO_TYPE = "yolov3"
YOLO_FRAMEWORK = "tf"
YOLO_V3_WEIGHTS = "yolov3.weights"
YOLO_CUSTOM_WEIGHTS = False
YOLO_IOU_LOSS_THRESH = 0.7
YOLO_STRIDES = [8, 16, 32]
YOLO_SCALES = [52, 26, 13]
YOLO_NUM_SCALES = 3
YOLO_ANCHOR_PER_SCALE = 3
YOLO_MAX_BBOX_PER_SCALE = 100
YOLO_ANCHORS_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
YOLO_LAYER_WITH_NAMES = True

# IMAGE size
IMAGE_SIZE = (416, 416)

# Dataset
# DIR_DATA is filled as a list to consider multiple dataset folders at same time
DIR_DATA = ["data/demo/"]
DIR_TRAIN = [d + "train/" for d in DIR_DATA]
DIR_VALID = [d + "valid/" for d in DIR_DATA]
DIR_TEST = [d + "test/" for d in DIR_DATA]
PATH_CLASSES = "data/ps_classes.txt"
PATH_ANCHORS = "data/yolo_anchors.txt"
PATH_WEIGHT = "data/yolov3_ps.h5"
PATH_DARKNET_WEIGHT = "data/yolov3.weights"

# TRAIN options
TRAIN_YOLO_TINY = False
TRAIN_SAVE_BEST_ONLY = True  # saves only best model according validation loss (True recommended)
TRAIN_SAVE_CHECKPOINT = False  # saves all best validated checkpoints in training process (may require a lot disk space) (False recommended)
TRAIN_ANNOT_PATH = [d + "_annotations.txt" for d in DIR_TRAIN]
TRAIN_CHECKPOINTS_FOLDER = "checkpoints"
TRAIN_MODEL_NAME = f"{YOLO_TYPE}_custom"
TRAIN_FROM_CHECKPOINT = False
TRAIN_TRANSFER = True
TRAIN_DATA_AUG = True
TRAIN_FREEZE_BODY = True
TRAIN_FREEZE_BATCH_SIZE = 32
TRAIN_UNFREEZE_BATCH_SIZE = 16  # note that more GPU memory is required after unfreezing the body
TRAIN_FREEZE_LR = 1e-3
TRAIN_UNFREEZE_LR = 1e-4
TRAIN_FREEZE_INIT_EPOCH = 0
TRAIN_FREEZE_END_EPOCH = 30
TRAIN_UNFREEZE_END_EPOCH = 60  # note that it is considered when TRAIN_FREEZE_BODY is True

# VAL options
VAL_ANNOT_PATH = [d + "_annotations.txt" for d in DIR_VALID]
VAL_DATA_AUG = False
VAL_BATCH_SIZE = 16
VAL_VALIDATION_USING = "TRAIN"  # note that when validation data does not exist, set it to TRAIN or None
VAL_VALIDATION_SPLIT = 0.2  # note that it will be used when VAL_VALIDATION_USING is TRAIN

# TEST options
TEST_ANNOT_PATH = [d + "_annotations.txt" for d in DIR_TEST]
TEST_BATCH_SIZE = 16
TEST_DATA_AUG = False
TEST_DETECTED_IMAGE_PATH = ""
TEST_SCORE_THRESHOLD = 0.3
TEST_IOU_THRESHOLD = 0.5

# LOG directory
LOG_DIR = "logs/"
LOG_DIR2 = ""
