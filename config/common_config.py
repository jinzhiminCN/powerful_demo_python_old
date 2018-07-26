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
