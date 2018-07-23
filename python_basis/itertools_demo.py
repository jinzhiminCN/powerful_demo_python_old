# -*- coding:utf-8 -*-
import itertools
import sys
import operator


def show_result(description, result):
    '''
    展示输出结果，包括输出列表和列表中元素的数量。
    :param description:输出数据的简单说明。
    :param result:输出数据的内容。
    :return: 无返回数据
    '''
    result_list = list(result)
    print("{0}结果：{1}，共{2}个元素。".format(description, result_list, len(result_list)))


def test_accumulate():
    '''
    测试迭代工具集中的累加方法。
    :return: 无返回数据
    '''
    # 元组累加
    result = itertools.accumulate([(10, 12, 5), (3, 4, 7)])
    show_result("元组累加", result)
    # 列表累加
    result = itertools.accumulate(range(10))
    show_result("列表累加", result)


def accumulate_source(iterable, func=operator.add):
    'Return running totals'
    # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = func(total, element)
        yield total


def test_chain():
    '''
    连接多个列表或者迭代器。
    :return: 无返回数据
    '''
    # 列表链接
    result = itertools.chain(range(3), range(4), [10, 3, 2, 1])
    show_result("列表链接", result)


def test_combinations():
    '''
    求列表或生成器中指定数目的元素不重复的所有组合。
    :return: 无返回数据
    '''
    # 元素不重复的所有组合
    result = itertools.combinations(range(4), 3)
    show_result("无重复组合", result)
    # 元素可重复的所有组合
    result = itertools.combinations_with_replacement(range(4), 3)
    show_result("可重复组合", result)


def test_compress():
    '''
    按照真值表筛选元素。
    :return:
    '''
    result = itertools.compress(range(5), (True, False, True, True, False))
    show_result("按照真值表筛选", result)


def test_islice():
    '''
    对迭代器进行切片
    :return:
    '''
    start, stop, step = 0, 9, 2
    result = itertools.islice(range(10), start, stop, step)
    show_result("迭代器切片", result)

    result = islice_source(range(10), start, stop, step)
    show_result("迭代器切片实现", result)


def islice_source(iterable, *args):
    s = slice(*args)
    start, stop, step = s.start or 0, s.stop or sys.maxsize, s.step or 1
    it = iter(range(start, stop, step))
    try:
        nexti = next(it)
    except StopIteration:
        # Consume *iterable* up to the *start* position.
        for i, element in zip(range(start), iterable):
            pass
        return
    try:
        for i, element in enumerate(iterable):
            if i == nexti:
                yield element
                nexti = next(it)
    except StopIteration:
        # Consume to *stop*.
        for i, element in zip(range(i + 1, stop), iterable):
            pass


def test_count():
    '''
    计数器
    :return:
    '''
    # 使用迭代方式遍历结果
    print("计数器结果：", end="")
    result = itertools.count(start=20, step=-1.5)
    for i in result:
        if i < 0:
            break
        print(i, end=' ')
    print("Next:", next(result))

    # 使用列表方式遍历结果，由于没有结束标记，无法退出迭代过程，不能直接输出。
    result = itertools.count(start=20, step=-1.5)
    # show_result("计数器", result)



if __name__ == "__main__":
    pass
    test_accumulate()
    test_chain()
    test_combinations()
    test_compress()
    test_islice()
    test_count()


