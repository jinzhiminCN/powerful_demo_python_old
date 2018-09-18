# -*- coding:utf-8 -*-

# ==============================================================================
# 操作系统os相关的常用操作集合。
# ==============================================================================

import os
import platform
from util.log_util import LoggerUtil

common_logger = LoggerUtil.get_common_logger()


class OsUtil(object):
    @staticmethod
    def makedirs(dir_path):
        """
        创建完整目录。
        :param dir_path: 目录名称
        :return:
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    @staticmethod
    def mk_full_dir(dir_path):
        """
        创建完整目录，如果目录不存在，则逐级创建对应的目录。
        :param dir_path: 目录名称
        :return:
        """
        if dir_path is None:
            return

        dir_path = dir_path.strip()
        if dir_path == "":
            return

        # 按照“/”划分目录层次
        dir_path_array = dir_path.split("/")
        current_dir_path = ""

        # 逐级生成目录
        for dir_path_item in dir_path_array:
            current_dir_path = os.path.join(current_dir_path, dir_path_item)
            OsUtil.mkdir(current_dir_path)
            current_dir_path += os.path.sep

    @staticmethod
    def mkdir(dir_path):
        """
        创建目录，如果目录不存在，则创建对应的目录。
        :param dir_path:
        :return:
        """
        if dir_path == "":
            return

        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    @staticmethod
    def exists(path):
        """
        文件或目录是否存在。
        :param path: 文件或目录的路径
        :return: bool类型
        """
        return os.path.exists(path)

    @staticmethod
    def is_window_sys():
        """
        是否为Windows系统。
        :return:
        """
        return platform.system() == "Windows"

    @staticmethod
    def is_linux_sys():
        """
        是否为Linux系统。
        :return:
        """
        return platform.system() == "Linux"

    @staticmethod
    def is_ios_sys():
        """
        是否为IOS系统 Mac或者osx。
        :return:
        """
        return platform.system() == "Darwin"

    @staticmethod
    def list_dir_num(dir_path):
        """
        列出目录下的所有目录结构和文件个数
        :return:
        """
        for dir_name, sub_dirs, filenames in os.walk(dir_path):
            if len(sub_dirs) > 0:
                common_logger.info("当前目录：{0}, 下级目录：{1}，共{2}个子目录。".format(dir_name, sub_dirs, len(sub_dirs)))
            if len(filenames) > 0:
                common_logger.info("当前目录：{0}, 共{1}个文件。".format(dir_name, len(filenames)))


