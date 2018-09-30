# -*- coding:utf-8 -*-

# ==============================================================================
# 测试configparser的相关方法。
# 配置文件的格式：
#   [section1]
#   name = nike
#   age = 20
#
#   [section2]
#   name = adidas
#   age = 18
# “[ ]”包含的内容为section，section下面为类似于key-value的配置内容；
# configparser 默认支持 ‘=’ ‘：’ 两种分隔。
# ==============================================================================
import configparser
import os
from util.log_util import LoggerUtil
from config.common_config import CONFIG_DIR

# 日志器
common_logger = LoggerUtil.get_common_logger()

# 配置文件路径
config_file_path = os.path.join(CONFIG_DIR, "config.ini")
config = configparser.ConfigParser()


def test_read_config():
    """
    读取配置信息。
    :return:
    """
    config_content = config.read(config_file_path)
    common_logger.info("config: {0}".format(config_content))
    # 所有section
    common_logger.info("sections: {0}".format(config.sections()))
    # section1下的所有key
    common_logger.info("options: {0}".format(config.options("section1")))
    # section1下的所有key和value
    common_logger.info("items: {0}".format(config.items("section1")))
    # section1下k1的value
    common_logger.info("get: {0}".format(config.get("section1", "name")))
    common_logger.info("get: {0}".format(config.getint("section1", "age")))

    # 把读在config里的内容删除section1
    config.remove_section("section1")
    # 在把config里的内容写进文件
    config.write(open(config_file_path, "w"))

    # 有没有section1，True或False
    common_logger.info(config.has_section("section1"))

    # 添加section
    config.add_section('section1')
    # 写入
    config.write(open(config_file_path, "w"))

    # 删除section
    config.remove_option("section2", "name")
    # 写入
    config.write(open(config_file_path, "w"))


if __name__ == "__main__":
    test_read_config()
    pass
