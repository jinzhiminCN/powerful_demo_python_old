# -*- coding:utf-8 -*-

# ==============================================================================
# 测试matplot的相关方法。
# http://whuhan2013.github.io/blog/2016/09/16/python-matplotlib-learn/
# ==============================================================================
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from image_process import pil_image_demo

# =========================== function ===========================


def test_plot_line_figure():
    """
    测试绘制曲线。使用全局绘图函数，此类 API 的好处是可以节省代码量，但是并不鼓励使用它处理复杂的图表。
    处理复杂图表时， matplotlib 面向对象 API 是一个更好的选择。
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


def test_plot_oop_figure():
    """
    使用面向对象的API，不创建一个全局实例，而是将新建实例的引用保存在 fig 变量中。
    如果想在图中新建一个坐标轴实例，只需要 调用 fig 实例的 add_axes 方法。
    :return:
    """
    # 曲线数据
    x = np.linspace(0, 5, 10)
    y = x**2

    # 绘制图像
    fig = plt.figure()

    # 图片1
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # left, bottom, width, height (range 0 to 1)
    axes.plot(x, y, 'r')
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_title('title')

    # 图片2
    axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3])  # inset axes
    # insert
    axes2.plot(y, x, 'g')
    axes2.set_xlabel('y')
    axes2.set_ylabel('x')
    axes2.set_title('insert title')

    plt.show()


def test_figure_layout():
    """
    测试图片默认布局。
    :return:
    """
    x = np.linspace(0, 5, 10)
    y = x ** 2

    fig, axes = plt.subplots(nrows=1, ncols=2)

    for ax in axes:
        ax.plot(x, y, 'r')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('title')

    fig.tight_layout()

    plt.show()


def test_figure_legend():
    """
    测试图例。
    :return:
    """
    x = np.linspace(0, 5, 10)

    # 绘图
    fig, ax = plt.subplots()

    # 普通图例
    # ax.plot(x, x ** 2, label="y = x**2")
    # ax.plot(x, x ** 3, label="y = x**3")
    # ax.legend(loc=2);  # upper left corner
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_title('title')

    # latex图例
    ax.plot(x, x ** 2, label=r"$y = \alpha^2$")
    ax.plot(x, x ** 3, label=r"$y = \alpha^3$")
    ax.legend(loc=2)  # upper left corner
    ax.set_xlabel(r'$\alpha$', fontsize=18)
    ax.set_ylabel(r'$y$', fontsize=18)
    ax.set_title('title')

    plt.show()


def test_figure_axis_range():
    """
    设置坐标轴的范围，可以使用 set_ylim 或是 set_xlim 方法
    或者 axis(‘tight’) 自动将坐标轴调整的紧凑。
    :return:
    """
    x = np.linspace(0, 5, 10)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].plot(x, x ** 2, x, x ** 3)
    axes[0].set_title("default axes ranges")

    axes[1].plot(x, x ** 2, x, x ** 3)
    axes[1].axis('tight')
    axes[1].set_title("tight axes")

    axes[2].plot(x, x ** 2, x, x ** 3)
    axes[2].set_ylim([0, 60])
    axes[2].set_xlim([2, 5])
    axes[2].set_title("custom axes range")

    plt.show()


def test_logarithmic_scale():
    """
    可以将轴的刻度设置成对数刻度，调用 set_xscale 与 set_yscale 设置刻度，参数选择 “log” 。
    :return:
    """
    x = np.linspace(0, 5, 10)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(x, x ** 2, x, np.exp(x))
    axes[0].set_title("Normal scale")

    axes[1].plot(x, x ** 2, x, np.exp(x))
    axes[1].set_yscale("log")
    axes[1].set_title("Logarithmic scale (y)")

    plt.show()


def test_figure_xyticks():
    """
    set_xticks 与 set_yticks 方法可以显示地设置标号的位置， set_xticklabels 与 set_yticklabels 为每一个标号设置符号。
    :return:
    """
    x = np.linspace(0, 5, 10)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(x, x ** 2, x, x ** 3, lw=2)

    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels([r'$\alpha$', r'$\beta$', r'$\gamma$', r'$\delta$', r'$\epsilon$'], fontsize=18)

    yticks = [0, 50, 100, 150]
    ax.set_yticks(yticks)
    ax.set_yticklabels(["$%.1f$" % y for y in yticks], fontsize=18)  # use LaTeX formatted labels

    plt.show()


def test_figure_grid():
    """
    grid 方法可以打开或关闭坐标网格线，也可以自定义网格的样式。
    :return:
    """
    x = np.linspace(0, 5, 10)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))

    # default grid appearance
    axes[0].plot(x, x ** 2, x, x ** 3, lw=2)
    axes[0].grid(True)

    # custom grid appearance
    axes[1].plot(x, x ** 2, x, x ** 3, lw=2)
    axes[1].grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)

    plt.show()


def test_figure_spines():
    """
    改变轴的属性。
    :return:
    """
    fig, ax = plt.subplots(figsize=(6, 2))

    ax.spines['bottom'].set_color('blue')
    ax.spines['top'].set_color('blue')

    ax.spines['left'].set_color('red')
    ax.spines['left'].set_linewidth(2)

    # turn off axis spine to the right
    ax.spines['right'].set_color("none")
    ax.yaxis.tick_left()  # only ticks on the left side

    plt.show()


def test_figure_origin_point():
    """
    设置坐标原点在（0，0）点。
    :return:
    """
    fig, ax = plt.subplots()

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))  # set position of x spine to x=0

    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))  # set position of y spine to y=0

    xx = np.linspace(-0.75, 1., 100)
    ax.plot(xx, xx ** 3)

    plt.savefig("1.png", dpi=600)
    plt.show()


def test_other_figure():
    """

    :return:
    """
    n = np.array([0, 1, 2, 3, 4, 5])
    xx = np.linspace(-1.5, 2, num=7)
    x = np.linspace(0, 5)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    axes[0][0].scatter(xx, xx + 0.25 * np.random.randn(len(xx)))
    axes[0][0].set_title("scatter")

    axes[0][1].step(n, n ** 2, lw=2)
    axes[0][1].set_title("step")

    axes[0][2].bar(n, n ** 2, align="center", width=0.5, alpha=0.5)
    axes[0][2].set_title("bar")

    axes[0][3].fill_between(x, x ** 2, x ** 3, color="green", alpha=0.5)
    axes[0][3].set_title("fill_between")

    axes[1][0].hist(n)
    axes[1][0].set_title("Default histogram")
    axes[1][0].set_xlim((min(n), max(n)))

    axes[1][1].hist(n, cumulative=True, bins=50)
    axes[1][1].set_title("Cumulative detailed histogram")
    axes[1][1].set_xlim((min(n), max(n)))

    labels = 'frogs', 'hogs', 'dogs', 'logs'
    sizes = 15, 20, 45, 10
    colors = 'yellowgreen', 'gold', 'lightskyblue', 'lightcoral'
    explode = 0, 0.1, 0, 0
    axes[1][2].pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=50)

    plt.show()


def test_polar_axis():
    """
    使用极坐标系
    :return:
    """
    # polar plot using add_axes and polar projection
    fig = plt.figure()
    ax = fig.add_axes([0.0, 0.0, .6, .6], polar=True)
    t = np.linspace(0, 2 * math.pi, 100)
    ax.plot(t, t, color='blue', lw=3)
    plt.show()


def test_subplot2grid():
    """
    将子图网格化布局。
    :return:
    """
    fig = plt.figure()
    ax1 = plt.subplot2grid((3,3), (0,0), colspan=3)
    ax2 = plt.subplot2grid((3,3), (1,0), colspan=2)
    ax3 = plt.subplot2grid((3,3), (1,2), rowspan=2)
    ax4 = plt.subplot2grid((3,3), (2,0))
    ax5 = plt.subplot2grid((3,3), (2,1))
    fig.tight_layout()

    plt.show()


def plt_image_test():
    """
    显示图像。
    :return:
    """
    im_nums = np.arange(8)
    im = np.zeros([8, 8])
    images = []

    for im_num in im_nums:
        im = im.copy()
        im[im_num, im_num] = im_num
        images.append(im)
    pil_image_demo.plt_images(images, 3, interpolation='nearest')


def plot_line_figure1():
    """
    https://blog.csdn.net/u012319493/article/details/80609905
    画折线。
    :return:
    """
    font1 = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 9,
    }

    x = []
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    for i in range(0, 1000):
        x.append(i)
        y1.append(random.randint(25, 40))
        y2.append(random.randint(20, 38))
        y3.append(random.randint(15, 36))
        y4.append(random.randint(12, 30))

    plt.figure()
    plt.plot(x, y1, color='green', label='y1')
    plt.plot(x, y2, color='red', label='y2')
    plt.plot(x, y3, color='skyblue', label='y3')
    plt.plot(x, y4, color='blue', label='y4')

    # 显示图例
    # plt.legend()
    # plt.legend(loc='upper right', prop=font1, frameon=False)
    plt.legend(loc='upper right', prop=font1, frameon=True)

    ax = plt.gca() # 获取到当前坐标轴信息
    ax.xaxis.set_ticks_position('top') # 将X坐标轴移到上面
    ax.invert_yaxis() # 反转Y坐标轴

    plt.xlabel('Iterations')
    plt.ylabel('Avg value')
    plt.show()


def plot_line_figure2():
    """
    画折线。
    :return:
    """
    font1 = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 9,
    }

    fig, ax = plt.subplots()

    x = []
    y0 = []
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    for i in range(0, 1000):
        x.append(i)
        y0.append(0)
        y1.append(random.randint(25, 40))
        y2.append(random.randint(20, 38))
        y3.append(random.randint(15, 36))
        y4.append(random.randint(12, 30))

    ax.plot(x, y0, color='black', linewidth=0)
    ax.plot(x, y1, color='green', label='y1')
    ax.plot(x, y2, color='red', label='y2')
    ax.plot(x, y3, color='skyblue', label='y3')
    ax.plot(x, y4, color='blue', label='y4')

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))  # set position of x spine to x=0

    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))  # set position of y spine to y=0
    # 显示图例
    # plt.legend()
    # plt.legend(loc='upper right', prop=font1, frameon=False)
    ax.legend(loc='upper right', prop=font1, frameon=True)

    ax.set_xlabel(r'Iterations', fontsize=9)
    ax.set_ylabel(r'Avg value', fontsize=9)
    ax.set_title('title')
    plt.show()


if __name__ == "__main__":
    pass
    # test_figure_xyticks()
    # test_logarithmic_scale()
    # test_figure_grid()
    # test_figure_spines()
    # test_figure_origin_point()
    # test_other_figure()
    # test_polar_axis()
    # test_subplot2grid()
    # plt_image_test()
    # test_plot_line_figure()
    plot_line_figure1()
    # plot_line_figure2()

