# -*- coding:utf-8 -*-

# ==============================================================================
# 装饰器的使用方法。
# ==============================================================================
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()


def decorate_without_param(func):
    """
    不带参数的简单装饰器。
    :param func:
    :return:
    """
    common_logger.info('running decorate{0}'.format(func))

    def decorate_inner():
        common_logger.info('running decorate_inner function')
        return func()
    return decorate_inner


def decorate_with_before_after(func):
    """
    包含函数运行前后内容的装饰器。
    :param func:
    :return:
    """
    def decorate_inner():
        common_logger.info('before decorate_inner function')
        ret = func()
        common_logger.info('after decorate_inner function')
        return ret
    return decorate_inner


def decorate_with_param(func):
    """
    包含函数参数的装饰器。
    :param func:
    :return:
    """
    def decorate_inner(*args, **kwargs):
        common_logger.info(type(args), type(kwargs))
        common_logger.info('args', args, 'kwargs', kwargs)
        return func(*args, **kwargs)
    return decorate_inner


def decorate_with_name(func):
    """
    包含name属性的装饰器。
    :param func:
    :return:
    """
    def decorate_inner(name, *args, **kwargs):
        common_logger.info(name, args, kwargs)
        return func(*args, **kwargs)
    return decorate_inner


def decorate_with_default_name(func):
    """
    包含默认name属性的装饰器。
    :param func:
    :return:
    """
    def decorate_inner(name="default", *args, **kwargs):
        common_logger.info(name, args, kwargs)
        common_logger.info(*args)
        common_logger.info(*kwargs)
        return func(name, *args, **kwargs)
    return decorate_inner


def outer(func):
    common_logger.info('enter outer {0}'.format(func))

    def wrapper():
        common_logger.info('running outer')
        func()
    return wrapper


def inner(func):
    common_logger.info('enter inner {0}'.format(func))

    def wrapper():
        common_logger.info('running inner')
        func()
    return wrapper


def trace(log_level):
    """
    装饰器带参数的情况。
    :param log_level:
    :return:
    """
    def impl_func(func):
        common_logger.info('{0} Implementing function: "{1}"'.format(log_level, func.__name__))
        return func
    return impl_func


@outer
@inner
def main():
    common_logger.info('running main')


@decorate_without_param
def func_1():
    common_logger.info('running func_1')


@decorate_with_before_after
def func_add():
    common_logger.info("1 + 2")
    return 1+2


@decorate_with_param
def func_2(*args, **kwargs):
    common_logger.info(args, kwargs)


@decorate_with_param
def func_3(name, *args, **kwargs):
    common_logger.info(name, args, kwargs)


@decorate_with_name
def func_4(name, *args, **kwargs):
    common_logger.info(name, args, kwargs)


@decorate_with_default_name
def func_5(name="some_name", *args, **kwargs):
    common_logger.info(name, args, kwargs)


@trace('[INFO]')
def print_msg(msg):
    common_logger.info(msg)


@trace('[DEBUG]')
def print_assert(expr):
    common_logger.info(expr)


def test_decorate():
    """
    测试装饰器。
    :return:
    """
    # common_logger.info("----------执行装饰器函数----------")
    # func_1()
    # common_logger.info("----------打印装饰器函数----------")
    # common_logger.info(func_1())
    # common_logger.info("----------打印装饰器函数名称----------")
    # common_logger.info(func_1)
    # common_logger.info("----------装饰器函数----------")
    # func_2('x', 'y', para1='1', para2='2')
    # func_3('x', 'y', para1='1', para2='2')
    # func_4('x', 'y', para1='1', para2='2')
    # func_4(nn='x', para1='1', para2='2')
    # func_4(name='x', para1='1', para2='2')
    # func_5(nn='x', para1='1', para2='2')
    # func_5('x2', name2='x', para1='1', para2='2')
    # func_add()
    # common_logger.info(func_add())
    print_msg('Hello, world!')


if __name__ == "__main__":
    test_decorate()
    pass

