# -*- coding:utf-8 -*-

# ==============================================================================
# 测试python相关的内建方法。
# globals()：内置的函数globals返回一个包含所有python解释器知道的变量名称的字典。
#
# ==============================================================================
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()


def test_locals(x=1, y=2):
    """
    测试locals函数。
    :param x:
    :param y:
    :return:
    """
    value = "hello"
    common_logger.info(locals())


def test_globals():
    """
    测试globals函数。
    :return:
    """
    common_logger.info(globals())
    common_logger.info("__doc__:{0}".format(__doc__))
    common_logger.info("__file__:{0}".format(__file__))


if __name__ == "__main__":
    # test_globals()
    test_locals()
    pass
