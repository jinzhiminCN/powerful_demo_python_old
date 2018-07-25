# -*- coding:utf-8 -*-

# https://docs.python.org/3/library/string.html#formatstrings


def str_format_test():
    """
    测试各种类型的字符串格式化。
    :return:
    """
    # Accessing arguments by position:
    print('{0}, {1}, {2}'.format('a', 'b', 'c'))
    print('{}, {}, {}'.format('a', 'b', 'c'))
    print('{2}, {1}, {0}'.format('a', 'b', 'c'))
    print('{2}, {1}, {0}'.format(*'abc'))
    print('{0}{1}{0}'.format('abra', 'cad'))

    # Accessing arguments by name:
    print('Coordinates: {latitude}, {longitude}'.format(latitude='37.24N', longitude='-115.81W'))
    coord = {'latitude': '37.24N', 'longitude': '-115.81W'}
    print('Coordinates: {latitude}, {longitude}'.format(**coord))

    # Aligning the text and specifying a width:
    print('{:<30}'.format('left aligned'))
    print('{:>30}'.format('right aligned'))
    print('{:^30}'.format('centered'))
    print('{:*^30}'.format(' centered '))

    # Using the comma as a thousands separator
    print('{:,}'.format(1234567890))


if __name__ == "__main__":
    pass
    str_format_test()