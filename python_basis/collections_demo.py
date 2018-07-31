# -*- coding:utf-8 -*-

# ==============================================================================
# 测试collections的相关方法。
# ==============================================================================
import time
import sys
from collections import namedtuple, deque, defaultdict, OrderedDict, Counter

# =========================== 全局常量 ===========================
default_time_format = '%Y-%m-%d %H:%M:%S'

# =========================== test function ===========================


def test_namedtuple():
    """
    测试namedtuple。
    :return:
    """
    # namedtuple是一个函数，它用来创建一个自定义的tuple对象。
    # 并且规定了tuple元素的个数，可以用属性而不是索引来引用tuple的某个元素。
    Point = namedtuple('Point', ['x', 'y'])
    point = Point(1, 2)
    print("Point:{0}, p.x={1}, p.y={2}.".format(point, point.x, point.y))
    print("is Point:{}".format(isinstance(point, Point)))
    print("is tuple:{}".format(isinstance(point, tuple)))


def test_dequeue():
    """
    测试dequeue的方法。
    :return:
    """
    q = deque(['a', 'b', 'c'])
    q.append('x')
    q.appendleft('y')
    print("DeQueue:".format(q))

    # rotate旋转
    q = deque(range(0, 10))
    for i in range(0, 10):
        q.rotate(1)
        print("{0}: {1}".format(i, q))

    q = deque(range(0, 10))
    for i in range(0, 10):
        q.rotate(i)
        print("{0}: {1}".format(i, q))

    # 走马灯
    fancy_loading = deque('>-------Hello-------------')

    while True:
        print('\r%s' % ''.join(fancy_loading))
        fancy_loading.rotate(1)
        sys.stdout.flush()
        time.sleep(0.08)


def test_counter():
    """
    测试counter的方法。
    :return:
    """
    c = Counter()
    for ch in 'programming':
        c[ch] += 1
    print(c)

    # 频率测试
    s = '''A Counter is a dict subclass for counting hashable objects. It is an unordered collection
        where elements are stored as dictionary keys and their counts are stored as dictionary values.
        Counts are allowed to be any integer value including zero or negative counts.
        The Counter class is similar to bags or multisets in other languages.'''.lower()

    c = Counter(s)
    # 获取出现频率最高的5个字符
    print("出现频率最高的5个字符：{0}".format(c.most_common(5)))


def test_orderedDict():
    """
    测试有序字典OrderedDict。
    :return:
    """
    items = [('a', 1), ('b', 2), ('c', 3)]

    # dict的Key是无序的
    d = dict(items)
    print("dict elements:{0}".format(d))

    # OrderedDict的Key是有序的
    od = OrderedDict(items)
    print("ordered dict elements:{0}".format(od))

    od = OrderedDict()
    od['z'] = 1
    od['y'] = 2
    od['x'] = 3
    list_dict = list(od.keys()) # 按照插入的Key的顺序返回
    print("ordered dict keys:{0}".format(list_dict))


def test_defaultdict():
    """
    在使用Python原生的数据结构dict的时候，如果用 d[key] 访问， 当指定的key不存在时，是会抛出KeyError异常的。
    但是，如果使用defaultdict，只要传入一个默认的工厂方法，那么请求一个不存在的key时， 便会调用这个工厂方法使用其结果来作为这个key的默认值。
    :return:
    """
    dd = defaultdict(lambda: 'N/A')
    dd['key1'] = 'abc'
    print("key1:", dd['key1'])
    print('key2:', dd['key2'])


if __name__ == "__main__":
    pass
    # test_namedtuple()
    # test_dequeue()
    # test_counter()
    # test_orderedDict()
    test_defaultdict()

