# -*- coding:utf-8 -*-

# ==============================================================================
# 日志的配置以及使用方法。
# ==============================================================================

import os
import sys
import re
import logging
import logging.config
from logging.handlers import TimedRotatingFileHandler
from logging.handlers import RotatingFileHandler
import yaml
import config.common_config as com_config
from config.common_config import COMMON_LOG_DIR


# =========================== 全局变量 ===========================
# common log directory

# 创建目录路径
if not os.path.exists(COMMON_LOG_DIR):
    os.makedirs(COMMON_LOG_DIR)

LOG_FILE = "application.log"
DATE_LOG = "date_log"

LOGGING_DICT = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                'format': '%(asctime)s [%(name)s:%(lineno)d] [%(levelname)s]- %(message)s'
            },
            'standard': {
                'format': '%(asctime)s [%(threadName)s:%(thread)d] [%(name)s:%(lineno)d] [%(levelname)s]- %(message)s'
            },
        },

        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "simple",
                "stream": "ext://sys.stdout"
            },

            "default": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "simple",
                "filename": os.path.join(COMMON_LOG_DIR, LOG_FILE),
                'mode': 'w+',
                "maxBytes": 1024*1024*5,  # 5 MB
                "backupCount": 20,
                "encoding": "utf8"
            },

            "date": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "level": "INFO",
                "formatter": "simple",
                "filename": os.path.join(COMMON_LOG_DIR, DATE_LOG),
                'when': "D",
                "backupCount": 20,
                "encoding": "utf8"
            },
        },

        "loggers": {
            "app": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": "no"
            }
        },

        "root": {
            'handlers': ['default', 'date'],
            'level': "INFO",
            'propagate': False
        }
    }


def test_dict_config():
    """
    通过字典配置日志。
    :return:
    """
    logging.config.dictConfig(LOGGING_DICT)

    log = logging.getLogger()
    print("print A")
    log.info("log B")


def test_yaml_config():
    """
    通过yaml文件配置日志。
    :return:
    """
    yaml_file = os.path.join(com_config.CONFIG_DIR, 'logging.yml')
    with open(yaml_file, 'r') as f_conf:
        dict_conf = yaml.load(f_conf)
    logging.config.dictConfig(dict_conf)

    logger = logging.getLogger('simpleExample')
    logger.debug('debug message')
    logger.info('info message')
    logger.info('通过yaml文件配置日志')
    logger.warning('warn message')
    logger.error('error message')
    logger.critical('critical message')


def test_basic_config():
    """
    配置日志基本信息。
    :return:
    """
    # 文件日志
    file_name = os.path.join(COMMON_LOG_DIR, LOG_FILE)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=file_name,
                        filemode='w')

    # console日志
    # 默认sys.stderr
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    # logging.getLogger('').setLevel(logging.DEBUG)

    # ''等同于root日志
    logging.getLogger('').addHandler(console)
    logger = logging.getLogger('')
    logger.info('info message')

    # 获取root日志
    logger = logging.getLogger()
    logger.info('info message')
    logger.warning("root default level is WARNING")


def test_logger():
    """
    测试日志类的使用方法。
    :return:
    """
    # 创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # ***************控制台日志处理器**************
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ***************日期循环日志处理器**************
    log_fmt = '%(asctime)s\tFile \"%(filename)s\",line %(lineno)s\t%(levelname)s: %(message)s'
    formatter = logging.Formatter(log_fmt)
    file_path = os.path.join(COMMON_LOG_DIR, "thread")
    log_file_handler = TimedRotatingFileHandler(filename=file_path, when="S", interval=1, backupCount=7)
    log_file_handler.suffix = "%Y-%m-%d_%H-%M.log"
    log_file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}.log$")
    log_file_handler.setFormatter(formatter)
    log_file_handler.setLevel(logging.DEBUG)
    logger.addHandler(log_file_handler)

    logger.info("test log")

if __name__ == '__main__':
    pass
    # test_dict_config()
    # test_yaml_config()
    # test_basic_config()
    test_logger()

