# -*- coding:utf-8 -*-

# ==============================================================================
# 测试EasyDict的使用方法。
# ==============================================================================
from easydict import EasyDict as edict
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()


def test_edict():
    """
    简单测试EasyDict
    :return:
    """
    # 使用dict初始化
    t_dict = edict({'foo': 3, 'bar': {'x': 1, 'y': 2}})
    common_logger.info("attr foo:{0}".format(t_dict.foo))
    common_logger.info("attr bar.x:{0}".format(t_dict.bar.x))
    common_logger.info("attr items:{0}".format(t_dict.items()))

    # 捕获异常
    try:
        common_logger.info("attr unset:{0}".format(t_dict.unset))
    except AttributeError as err:
        common_logger.info(err)

    # 设置属性
    t_dict.unset = "unset_value001"
    common_logger.info("attr unset:{0}".format(t_dict.unset))

    # 使用赋值语句初始化
    t_dict2 = edict(full="!value!", top=123)
    common_logger.info("attr full:{0}".format(t_dict2.full))
    common_logger.info("attr top:{0}".format(t_dict2.top))


if __name__ == "__main__":
    test_edict()
    pass

