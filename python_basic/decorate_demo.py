# -*- coding:utf-8 -*-

# ==============================================================================
# 装饰器的使用方法。
# ==============================================================================


def decorate_without_param(func):
    """
    不带参数的简单装饰器。
    :param func:
    :return:
    """
    print('running decorate', func)

    def decorate_inner():
        print('running decorate_inner function')
        return func()
    return decorate_inner


def decorate_with_before_after(func):
    """
    包含函数运行前后内容的装饰器。
    :param func:
    :return:
    """
    def decorate_inner():
        print('before decorate_inner function')
        ret = func()
        print('after decorate_inner function')
        return ret
    return decorate_inner


def decorate_with_param(func):
    def decorate_inner(*args, **kwargs):
        print(type(args), type(kwargs))
        print('args', args, 'kwargs', kwargs)
        return func(*args, **kwargs)
    return decorate_inner


def decorate_with_name(func):
    def decorate_inner(name, *args, **kwargs):
        print(name, args, kwargs)
        return func(*args, **kwargs)
    return decorate_inner


def decorate_with_default_name(func):
    def decorate_inner(name="default", *args, **kwargs):
        print(name, args, kwargs)
        print(*args)
        print(*kwargs)
        return func(name, *args, **kwargs)
    return decorate_inner


def outer(func):
    print('enter outer', func)

    def wrapper():
        print('running outer')
        func()
    return wrapper


def inner(func):
    print('enter inner', func)

    def wrapper():
        print('running inner')
        func()
    return wrapper


@outer
@inner
def main():
    print('running main')


@decorate_without_param
def func_1():
    print('running func_1')


@decorate_with_before_after
def func_add():
    print("1 + 2")
    return 1+2


@decorate_with_param
def func_2(*args, **kwargs):
    print(args, kwargs)


@decorate_with_param
def func_3(name, *args, **kwargs):
    print(name, args, kwargs)


@decorate_with_name
def func_4(name, *args, **kwargs):
    print(name, args, kwargs)


@decorate_with_default_name
def func_5(name="some_name", *args, **kwargs):
    print(name, args, kwargs)


def test_decorate():
    """
    测试装饰器。
    :return:
    """
    # print("----------执行装饰器函数----------")
    # func_1()
    # print("----------打印装饰器函数----------")
    # print(func_1())
    # print("----------打印装饰器函数名称----------")
    # print(func_1)
    # print("----------装饰器函数----------")
    # func_2('x', 'y', para1='1', para2='2')
    # func_3('x', 'y', para1='1', para2='2')
    func_4('x', 'y', para1='1', para2='2')
    # func_4(nn='x', para1='1', para2='2')
    # func_4(name='x', para1='1', para2='2')
    # func_5(nn='x', para1='1', para2='2')
    # func_5('x2', name2='x', para1='1', para2='2')
    # func_add()
    # print(func_add())


if __name__ == "__main__":
    test_decorate()
    pass

