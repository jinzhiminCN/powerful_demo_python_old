# -*- coding:utf-8 -*-

# ==============================================================================
# 测试线性回归的相关方法。
# ==============================================================================
import numpy as np
from sklearn import linear_model
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()


def least_square_approach(x_array, y_array):
    """
    根据公式计算最小二乘法。
    x theta = y
    theta = (xTx)^(-1)xTy
    :param x_array: ndarray类型，必须是一维数据或二位数据。
    :param y_array: ndarray类型，必须是一维数据
    :return:
    """
    x_ndim = x_array.ndim
    y_ndim = y_array.ndim

    if x_ndim > 2:
        raise AttributeError("x_array 维度必须为一维或二维")
    if y_ndim > 2:
        raise AttributeError("y_array 维度必须为一维或二维")

    # 如果x_array是一维数据，需要转换为二维
    if x_ndim == 1:
        x_array = x_array.reshape((x_array.shape[0], 1))
    # 如果x_array是一维数据，y_array需要转换为二维
    if y_ndim == 1:
        y_array = y_array.reshape((-1, 1))
    elif y_ndim == 2:
        if y_array.shape[1] > 1:
            raise AttributeError("y_array 第二维只能有一个元素")

    # common_logger.info("x_array:\n{0}".format(x_array))
    # common_logger.info("y_array:\n{0}".format(y_array))
    # common_logger.info("x_array.T:\n{0}".format(x_array.transpose()))

    # 计算xTx
    xTx = np.matmul(x_array.transpose(), x_array)
    # common_logger.info("xTx:\n{0}".format(xTx))

    xTx_det = np.linalg.det(xTx)
    if xTx_det == 0.0:
        print("xTx不能求逆矩阵")
        return
    xTx_inv = np.linalg.inv(xTx)

    theta = np.matmul(xTx_inv, np.matmul(x_array.transpose(), y_array))
    common_logger.info("theta:\n{0}".format(theta))


def least_square_mat(x_list, y_list):
    """
    使用matrix计算最小二乘法。
    :param x_list:
    :param y_list:
    :return:
    """
    # 将数组转换为矩阵
    x_mat = np.mat(x_list)
    y_mat = np.mat(y_list)

    # common_logger.info("x_mat:\n{0}".format(x_mat))
    # common_logger.info("y_mat:\n{0}".format(y_mat))
    # common_logger.info("x_mat.T:\n{0}".format(x_mat.T))

    xTx = x_mat.T * x_mat  # 计算xTx的
    common_logger.info("xTx:\n{0}".format(xTx))

    if np.linalg.det(xTx) == 0.0:
        print('xTx不能求逆矩阵')
        return

    common_logger.info("xTx.I:\n{0}".format(xTx.I))

    theta = xTx.I * (x_mat.T * y_mat)
    common_logger.info("theta:\n{0}".format(theta))


def linear_regression_sk(x, y):
    """
    使用sklearn的LinearRegression。
    :param x:
    :param y:
    :return:
    """
    linear_estimator = linear_model.LinearRegression()
    linear_estimator.fit(x, y)

    common_logger.info("coef:{0}, intercept:{1}".format(linear_estimator.coef_, linear_estimator.intercept_))


def test_linear_reg():
    """
    测试线性回归。
    :return:
    """
    train_x = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59,
                          2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
    train_y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53,
                          1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])

    # 一维求解
    least_square_approach(train_x, train_y)

    train_x = train_x.reshape([-1, 1])
    train_y = train_y.reshape([-1, 1])
    linear_regression_sk(train_x.tolist(), train_y.tolist())
    least_square_mat(train_x, train_y)

    # 二维求解
    train_x2 = np.ones(train_x.shape)
    train_x3 = np.hstack((train_x, train_x2))
    least_square_approach(train_x3, train_y)
    least_square_mat(train_x3, train_y)

if __name__ == "__main__":
    test_linear_reg()
    pass

