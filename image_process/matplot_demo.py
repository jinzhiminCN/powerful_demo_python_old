# -*- coding:utf-8 -*-

# ==============================================================================
# 测试matplot的相关方法。
# http://whuhan2013.github.io/blog/2016/09/16/python-matplotlib-learn/
# ==============================================================================

import matplotlib.pyplot as plt
import numpy as np

# =========================== test function ===========================


def plot_line():
    """
    测试绘制曲线。
    :return:
    """
    # 曲线数据
    x = np.linspace(0, 5, 10)
    y = x**2

    # 绘制曲线
    plt.figure()
    plt.plot(x, y, 'r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('y = x**2')
    plt.show()

    # 绘制子图
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(x, y, 'r--')
    plt.subplot(1, 2, 2)
    plt.plot(y, x, 'g*-')
    plt.show()


if __name__ == "__main__":
    pass
    plot_line()