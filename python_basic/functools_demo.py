# -*- coding:utf-8 -*-

# ==============================================================================
# functools的使用方法。
# update_wrapper(wrapper, wrapped)的工作很简单，就是用参数wrapped表示的函数对象的一些属性
# （如：__name__、 __doc__）覆盖参数wrapper表示的函数对象的相应属性。
# functools.wraps装饰器就是调用update_wrapper方法来修改被修饰函数的属性为原始函数的属性。
# ==============================================================================
import functools
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()


def my_decorator(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        common_logger.info('Calling decorated function')
        return f(*args, **kwargs)
    return wrapper


@my_decorator
def example():
    """
    Docstring
    """
    common_logger.info('Called example function')


def test_wraps():
    """
    测试wraps装饰器。
    :return:
    """
    example()
    common_logger.info(example.__name__)
    common_logger.info(example.__doc__)


def add1(a, b, c):
    return a + b + c


def add2(*args, **kwargs):
    common_logger.info("args的值为：{0}".format(args))
    common_logger.info("kwargs的值为：{0}".format(kwargs))


def test_partial():
    """
    测试partial函数。
    :return:
    """
    common_logger.info("add result:{0}".format(add1(1, 2, 3)))
    add_plus = functools.partial(add1, 100)
    common_logger.info("partial(add, 100) result:{0}".format(add_plus(5, 6)))
    add_plus = functools.partial(add1, 100, 200)
    common_logger.info("partial(add, 100, 200) result:{0}".format(add_plus(5)))
    new_add = functools.partial(add2, 1, 2, 3, a=100, b=200, c=300)
    common_logger.info("new_add result:{0}".format(new_add(4, 5, e=400, f=500)))


if __name__ == "__main__":
    # test_wraps()
    test_partial()
    pass
