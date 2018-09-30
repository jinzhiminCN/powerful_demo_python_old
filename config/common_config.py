# -*- encoding:utf-8 -*-

# ==============================================================================
# 通用的配置信息。
# ==============================================================================

import os

# project directory
# 需要保证config模块的目录结构不变，才能得到项目的目录。
PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))

# config directory
CONFIG_DIR = os.path.join(PROJECT_DIR, "config")

# resource directory
RESOURCE_DIR = os.path.join(PROJECT_DIR, "resource")

# common log directory
COMMON_LOG_DIR = os.path.join(PROJECT_DIR, 'logs')

# mnist data dir
MNIST_DIR = "E:/test_data/mnist"

# tensorflow model directory
TF_MODEL_LOGS_DIR = 'E:/logs/tensorflow_logs/'

# model checkpoint directory
TF_MODEL_CHECKPOINT_DIR = 'E:/logs/checkpoint/'

# redis ip and port
REDIS_ADDRESS = ''
