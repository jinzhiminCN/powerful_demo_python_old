# -*- coding:utf-8 -*-

# ==============================================================================
# 测试subprocess.Popen操作的文件。
# ==============================================================================


def add():
    """
    测试add操作。
    :return:
    """
    x = input('input x:').strip()
    print('\n')
    y = input('input y:').strip()
    print('\n')
    print('the result is ', int(x)+int(y))


if __name__ == "__main__":
    add()
