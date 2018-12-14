# -*- coding:utf-8 -*-

# ==============================================================================
# 测试str.format以及相关的格式化方法。
# ==============================================================================
# https://docs.python.org/3/library/string.html#formatstrings
# ==============================================================================
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()


def str_format_test():
    """
    测试各种类型的字符串格式化。
    :return:
    """
    # Accessing arguments by position:
    common_logger.info('{0}, {1}, {2}'.format('a', 'b', 'c'))
    common_logger.info('{}, {}, {}'.format('a', 'b', 'c'))
    common_logger.info('{2}, {1}, {0}'.format('a', 'b', 'c'))
    common_logger.info('{2}, {1}, {0}'.format(*'abc'))
    common_logger.info('{0}{1}{0}'.format('abra', 'cad'))

    # Accessing arguments by name:
    common_logger.info('Coordinates: {latitude}, {longitude}'.format(latitude='37.24N', longitude='-115.81W'))
    coord = {'latitude': '37.24N', 'longitude': '-115.81W'}
    common_logger.info('Coordinates: {latitude}, {longitude}'.format(**coord))

    # Replacing % +f, % -f, and % f and specifying a sign:
    common_logger.info('{:+f}; {:+f}'.format(3.14, -3.14))
    common_logger.info('{: f}; {: f}'.format(3.14, -3.14))
    common_logger.info('{:-f}; {:-f}'.format(3.14, -3.14))
    common_logger.info('{:.2f}; {:-.2f}'.format(3.1415, -3.1415))

    # Aligning the text and specifying a width:
    common_logger.info('{:<30}'.format('left aligned'))
    common_logger.info('{:>30}'.format('right aligned'))
    common_logger.info('{:^30}'.format('centered'))
    common_logger.info('{:*^30}'.format(' centered '))

    # Using the comma as a thousands separator
    common_logger.info('{:,}'.format(1234567890))
    common_logger.info('{:,.2f}'.format(1234567890.01))

    # Replacing %x and %o and converting the value to different bases
    common_logger.info("int: {0:d};  hex: {0:x};  oct: {0:o};  bin: {0:b}".format(42))
    # with 0x, 0o, or 0b as prefix:
    common_logger.info("int: {0:d};  hex: {0:#x};  oct: {0:#o};  bin: {0:#b}".format(42))


def output_format_test():
    """
    格式化输出测试，使用%。
    :return:
    """
    # % o —— oct 八进制
    # % d —— dec 十进制
    # % x —— hex 十六进制
    common_logger.info('%o, %d, %x' % (20, 20, 20))

    # float format
    # % f ——保留小数点后面六位有效数字
    # 　　 % .3f，保留3位小数位
    # % e ——保留小数点后面六位有效数字，指数形式输出
    # 　　 % .3e，保留3位小数位，使用科学计数法
    # % g ——在保证六位有效数字的前提下，使用小数方式，否则使用科学计数法
    # 　　 % .3g，保留3位有效数字，使用小数或科学计数法
    common_logger.info('%f' % 1.11)
    common_logger.info('%.1f' % 1.11)
    common_logger.info('%e' % 1.11)
    common_logger.info('%e' % 1.11)
    common_logger.info('%g' % 1111.1111)
    common_logger.info('%.7g' % 1111.1111)
    common_logger.info('%.2g' % 1111.1111)

    # % s
    # % 10s——右对齐，占位符10位
    # % -10s——左对齐，占位符10位
    # % .2s——截取2位字符串
    # % 10.2s——10位占位符，截取两位字符串
    common_logger.info('%s' % 'hello world')
    common_logger.info('%20s' % 'hello world')
    common_logger.info('%.2s' % 'hello world')
    common_logger.info('%10.2s' % 'hello world')
    common_logger.info('%-10.2s' % 'hello world')


if __name__ == "__main__":
    pass
    # str_format_test()
    output_format_test()
