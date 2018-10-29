# -*- coding:utf-8 -*-

# ==============================================================================
# 测试tqdm的基本用法。
# tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，
# 用户只需要封装任意的迭代器 tqdm(iterator)
# ==============================================================================
import os
import time
from tqdm import tqdm, tnrange, tqdm_notebook
from config.common_config import *
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()


def test_range_tqdm():
    """
    测试range的tqdm。
    :return:
    """
    for i in tqdm(range(10000)):
        time.sleep(0.01)


def test_list_tqdm():
    """
    测试list的tqdm。
    :return:
    """
    pro_bar = tqdm(["a", "b", "c", "d"])
    for char in pro_bar:
        pro_bar.set_description("Processing {0}".format(char))
        time.sleep(0.5)


def test_dir_tqdm():
    """
    测试文件目录的tqdm。
    :return:
    """
    resource_dir = RESOURCE_DIR
    if os.path.exists(resource_dir):
        for file in tqdm(os.listdir(resource_dir)):
            time.sleep(1)
            print(file)
            common_logger.info(file)


def test_tnrange():
    """
    测试tnrange。Only IPython/Jupyter Notebook.
    :return:
    """
    for i in tnrange(range(4), desc='1st loop'):
        for j in tnrange(range(100), desc='2nd loop'):
            time.sleep(0.1)


def test_tqdm_notebook():
    """
    测试tqdm_notebook。Only IPython/Jupyter Notebook.
    :return:
    """
    for i in tqdm_notebook(range(4), desc='1st loop'):
        for j in tqdm_notebook(range(100), desc='2nd loop'):
            time.sleep(0.1)


if __name__ == "__main__":
    # test_range_tqdm()
    # test_list_tqdm()
    # test_dir_tqdm()
    # test_tnrange()
    test_tqdm_notebook()
    pass
