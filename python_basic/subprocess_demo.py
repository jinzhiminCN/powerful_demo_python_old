# -*- coding:utf-8 -*-

# ==============================================================================
# 测试subprocess的相关方法。
# ==============================================================================
import subprocess
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()


def test_call():
    """
    测试subprocess的call方法。
    :return:
    """
    result = subprocess.call('whoami')
    common_logger.info(result)

    result = subprocess.call('ls -l', shell=True)
    common_logger.info(result)

    result = subprocess.call('dir', shell=True)
    common_logger.info(result)


def test_check_call():
    """
    执行命令，如果执行状态码是0，则返回0，否则抛异常。
    :return:
    """
    # result = subprocess.check_call('ls -l', shell=True)
    # common_logger.info(result)

    result = subprocess.check_call('ipconfig', shell=True)
    common_logger.info(result)


def test_popen():
    """
    测试Popen方法。
    :return:
    """
    # result = subprocess.Popen('ipconfig', shell=True, stdout=subprocess.PIPE).stdout.read()
    # common_logger.info(str(result, "gbk"))

    status, result = subprocess.getstatusoutput("ipconfig")
    common_logger.info("{0}, {1}".format(status, result))


if __name__ == "__main__":
    # test_call()
    # test_check_call()
    test_popen()
    pass
