# -*- coding: utf-8 -*-

# ==============================================================================
# 日志log相关的工具。
# ==============================================================================

import os
import logging
import sys
import logging.config
import config.common_config as com_config

log_path = com_config.COMMON_LOG_DIR
if not os.path.exists(log_path):
    os.makedirs(log_path)

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '[%(levelname)s] %(asctime)s [%(name)s:%(lineno)d] - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
    },
    'handlers': {
        'common': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'verbose',
            'filename': os.path.join(log_path, 'common.log'),
            'maxBytes': 1024 * 1024 * 10,
            'backupCount': 200,
            'encoding': 'utf8'
        },
        'warn': {
            'level': 'WARNING',
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'verbose',
            'filename': os.path.join(log_path, 'warning.log'),
            'maxBytes': 1024 * 1024 * 10,
            'backupCount': 200,
            'encoding': 'utf8'
        },
        'error': {
            'level': 'ERROR',
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'verbose',
            'filename': os.path.join(log_path, 'error.log'),
            'maxBytes': 1024 * 1024 * 10,
            'backupCount': 200,
            'encoding': 'utf8'
        },
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose'
        },
    },
    'loggers': {
        'common': {
            'handlers': ['console', 'common'],
            'level': 'INFO',
            'propagate': False
        },
        'warning': {
            'handlers': ['warn', 'console'],
            'level': 'WARNING',
            'propagate': False
        },
        'error': {
            'handlers': ['error', 'console'],
            'level': 'ERROR',
            'propagate': False
        },
    }
}


logging.config.dictConfig(LOGGING)


# 通用日志
common_logger = logging.getLogger("common")


class LoggerUtil(object):
    @staticmethod
    def get_common_logger():
        return common_logger

