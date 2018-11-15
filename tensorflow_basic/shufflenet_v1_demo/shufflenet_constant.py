# -*- coding:utf-8 -*-

# ==============================================================================
# shufflenet网络的常量。
# ==============================================================================
import os

# tiny_imagenet_dir
TINY_IMAGENET_DIR = "E:/test_data/tiny-imagenet-200/"
TFRECORD_DIR = os.path.join(TINY_IMAGENET_DIR, "tfrecord")
TRAIN_TFRECORD_PATH = os.path.join(TFRECORD_DIR, 'train.tfrecords')
VAL_TFRECORD_PATH = os.path.join(TFRECORD_DIR, 'val.tfrecords')


IMAGE_SIZE = 56
NUM_CLASSES = 200

BATCH_NORM_MOMENTUM = 0.1

N_SHUFFLE_UNITS = (1, 3, 1)

FIRST_STRIDE = 1

# optimizer settings
MOMENTUM = 0.9
USE_NESTEROV = True
LR_REDUCE_FACTOR = 0.1

SHUFFLE_BUFFER_SIZE = 10000
PREFETCH_BUFFER_SIZE = 1000
NUM_THREADS = 4


