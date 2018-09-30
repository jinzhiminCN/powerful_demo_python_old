# -*- coding:utf-8 -*-

# ==============================================================================
# 测试argparse的相关方法。
# ==============================================================================
import argparse

# 创建一个解析对象
parser = argparse.ArgumentParser()


def simple_arg():
    """
    简单参数解析。
    :return:
    """
    # 向解析对象中添加要关注的命令行参数和选项
    parser.add_argument('echo')
    parser.add_argument('simple')
    # 解析对象的参数内容
    args = parser.parse_args()
    print(args)
    print(args.echo)


def square_args():
    """
    计算平方值的参数解析。
    :return:
    """
    global parser
    parser = argparse.ArgumentParser(description='default_args')
    parser.add_argument("square", type=int, default=3,
                        help="display a square of a given number")
    parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2],
                        help="increase output verbosity")
    args = parser.parse_args()
    answer = args.square**2
    if args.verbosity == 2:
        print("the square of {0} equals {1}".format(args.square, answer))
    elif args.verbosity == 1:
        print("{}^2 == {}".format(args.square, answer))
    else:
        print(answer)


def detail_arg():
    """
    详细的参数解析。
    :return:
    """
    global parser
    parser = argparse.ArgumentParser(description='this is a description')
    # 该参数能接受的值只能来自某几个值候选值中，除此之外会报错
    parser.add_argument('dest_file', choices=['test1', 'test2'])
    # nargs - 指定这个参数后面的value有多少个，默认为1。
    # nargs 还可以用'*'来表示如果有该位置参数输入的话，之后所有的输入都将作为该位置参数的值；
    # ‘+’表示读取至少1个该位置参数。'?'表示该位置参数要么没有，要么就只要一个。
    parser.add_argument('input_file', nargs='+')
    parser.add_argument('--key', '-k', default="some_key", help='haha_key')
    # action=store_true表示是否为true或false
    parser.add_argument('--ver', '-v', action='store_true', help='haha_ver')
    # 参数是必需的，并且类型为int，输入其它类型会报错
    parser.add_argument('--int', '-i', required=True, type=int)
    # dest选项表示value解析出来后放到哪个属性中
    parser.add_argument('--world', choices=['test1', 'test2'], dest='world')
    # 将变量以标签-值的字典形式存入args字典
    args = parser.parse_args()
    print(args)
    if args.ver:
        print("True")
    else:
        print("False")


if __name__ == "__main__":
    pass
    # simple_arg()
    detail_arg()
