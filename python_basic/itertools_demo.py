# -*- coding:utf-8 -*-

# ==============================================================================
# 测试itertools工具类中的各种迭代方法。
# ==============================================================================

import itertools
import sys
import operator


def show_result(description, result):
    """
    展示输出结果，包括输出列表和列表中元素的数量。
    :param description:输出数据的简单说明。
    :param result:输出数据的内容。
    :return: 无返回数据
    """
    result_list = list(result)
    print("{0}结果：{1}，共{2}个元素。".format(description, result_list, len(result_list)))


def test_accumulate():
    """
    测试迭代工具集中的累加方法。
    :return: 无返回数据
    """
    # 元组累加
    result = itertools.accumulate([(10, 12, 5), (3, 4, 7)])
    show_result("元组累加", result)
    # 列表累加
    result = itertools.accumulate(range(10))
    show_result("列表累加", result)


def test_chain():
    """
    连接多个列表或者迭代器。
    :return: 无返回数据
    """
    # 列表链接
    result = itertools.chain(range(3), range(4), [10, 3, 2, 1])
    show_result("列表链接", result)


def test_combinations():
    """
    求列表或生成器中指定数目的元素不重复的所有组合。
    :return: 无返回数据
    """
    # 元素不重复的所有组合
    result = itertools.combinations(range(4), 3)
    show_result("无重复组合", result)
    # 元素可重复的所有组合
    result = itertools.combinations_with_replacement(range(4), 3)
    show_result("可重复组合", result)


def test_permutations():
    """
    产生指定数目的元素的所有排列(顺序有关)。
    :return:
    """
    result = itertools.permutations(range(4), 3)
    show_result("全排列", result)


def test_product():
    """
    产生多个列表和迭代器的笛卡尔积。
    :return:
    """
    result = itertools.product('ABC', range(3))
    show_result("全排列", result)

    result = itertools.product('ABC', "AAA")
    show_result("全排列", result)

    elements = "ABC"
    result = itertools.product(elements, elements)
    result = itertools.product(result, elements)
    show_result("全排列", result)


def test_compress():
    """
    按照真值表筛选元素。
    :return:
    """
    result = itertools.compress(range(5), (True, False, True, True, False))
    show_result("按照真值表筛选", result)


def test_islice():
    """
    对迭代器进行切片。
    :return:
    """
    start, stop, step = 0, 9, 2
    result = itertools.islice(range(10), start, stop, step)
    show_result("迭代器切片", result)

    result = islice_builtin(range(10), start, stop, step)
    show_result("迭代器切片实现", result)


def test_count():
    """
    计数器测试。
    :return:
    """
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
    # show_result("计数器", result)  #不能直接使用
    slice_result = itertools.islice(result, 0, 10, 1)
    show_result("计数器切片", slice_result)


def test_cycle():
    """
    循环指定的列表和迭代器。
    :return:
    """
    result = itertools.cycle('ABC')
    slice_result = itertools.islice(result, 0, 10, 1)
    show_result("循环器切片", slice_result)


def test_repeat():
    """
    简单的生成一个拥有指定数目元素的迭代器。
    :return:
    """
    result = itertools.repeat(10, 5)
    show_result("重复器", result)


def test_dropwhile():
    """
    按照真值函数丢弃掉列表和迭代器前面的元素。
    :return:
    """
    result = itertools.dropwhile(lambda e: e < 5, range(10, 0, -1))
    show_result("丢弃符合条件的前几项", result)

    result = itertools.dropwhile(lambda e: e < 5, list(range(10)) + list(range(10, 0, -1)))
    show_result("丢弃符合条件的前几项", result)


def test_takewhile():
    """
    保留元素直至真值函数值为假。
    :return:
    """
    result = itertools.takewhile(lambda e: e < 5, range(10, 0, -1))
    show_result("保留符合条件的前几项", result)

    result = itertools.takewhile(lambda e: e < 5, range(10))
    show_result("保留符合条件的前几项", result)


def test_filterfalse():
    """
    保留对应真值为False的元素。
    :return:
    """
    result = itertools.dropwhile(lambda e: e < 5, range(10))
    show_result("按条件过滤", result)


def test_groupby():
    """
    按照分组函数的值对元素进行分组。
    :return:
    """
    result = itertools.groupby(range(10), lambda x: x < 5 or x > 8)
    for condition, numbers in result:
        print(condition, list(numbers))


def test_starmap():
    """
    标记是否符合条件，类似map。
    :return:
    """
    result = itertools.starmap(str.islower, 'aBCDefGhI')
    show_result("标记是否符合条件", result)

    result = map(str.islower, 'aBCDefGhI')
    show_result("标记是否符合条件", result)


def test_zip_longest():
    """
    类似于zip，不过以较长的列表和迭代器的长度为准。
    :return:
    """
    result = itertools.zip_longest(range(3), range(5))
    show_result("构造最长zip对", result)

    result = zip(range(3), range(5))
    show_result("构造zip对", result)


def test_tee():
    """
    生成指定数目的迭代器。
    :return:
    """
    result = itertools.tee(range(5), 3)
    show_result("构造tee结果", result)
    for eles in result:
        print(list(eles))


# =========================== python builtin function ===========================

def accumulate_builtin(iterable, func=operator.add):
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


def islice_builtin(iterable, *args):
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


def cycle_builtin(iterable):
    # cycle('ABCD') --> A B C D A B C D A B C D ...
    saved = []
    for element in iterable:
        yield element
        saved.append(element)
    while saved:
        for element in saved:
            yield element


def dropwhile_builtin(predicate, iterable):
    # dropwhile(lambda x: x<5, [1,4,6,4,1]) --> 6 4 1
    iterable = iter(iterable)
    for x in iterable:
        if not predicate(x):
            yield x
            break
    for x in iterable:
        yield x


def filterfalse_builtin(predicate, iterable):
    # filterfalse(lambda x: x%2, range(10)) --> 0 2 4 6 8
    if predicate is None:
        predicate = bool
    for x in iterable:
        if not predicate(x):
            yield x


if __name__ == "__main__":
    pass
    # test_accumulate()
    # test_chain()
    # test_combinations()
    # test_permutations()
    # test_product()
    # test_compress()
    # test_islice()
    # test_count()
    # test_cycle()
    # test_repeat()
    # test_dropwhile()
    # test_takewhile()
    # test_filterfalse()
    # test_groupby()
    test_starmap()
    test_zip_longest()
    test_tee()