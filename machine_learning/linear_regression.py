# -*- coding:utf-8 -*-

# ==============================================================================
# 测试线性回归的相关方法。
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()


class LinearRegression(object):
    def __init__(self, gradient_descent_type=1):
        """
        初始化
        """
        self.w = None
        self.b = None
        # 1: batch gradient descent, 2: random gradient descent,
        self.gradient_descent_type = gradient_descent_type

    def loss(self, x, y):
        """
        计算损失函数。
        :param x:
        :param y:
        :return:
        """
        # 如果参数是一维数据，则转换为二维
        if x.shape == 1:
            x = x.reshape([1, -1])

        # 输入数据行数
        num_train = x.shape[0]

        h = x.dot(self.w) + self.b
        loss = 0.5 * np.sum(np.square(h - y)) / num_train

        dw = x.T.dot((h - y)) / num_train
        db = np.sum((h - y)) / num_train

        return loss, dw, db

    def train(self, x, y, learn_rate=0.001, iters=10000):
        """
        训练数据。
        :param x:
        :param y:
        :param learn_rate:
        :param iters:
        :return:
        """
        num_feature = x.shape[1]
        self.w = np.zeros((num_feature, 1))
        self.b = 0
        loss_list = []
        train_num = len(x)

        for i in range(iters):
            # 批量梯度下降
            if self.gradient_descent_type == 1:
                loss, dw, db = self.loss(x, y)
                loss_list.append(loss)
                self.w += -learn_rate * dw
                self.b += -learn_rate * db
            # 随机梯度下降
            elif self.gradient_descent_type == 2:
                for train_i in range(train_num):
                    loss, dw, db = self.loss(x[train_i], y[train_i])
                    loss_list.append(loss)
                    self.w += -learn_rate * dw
                    self.b += -learn_rate * db

            if i % 500 == 0:
                print('iters = {0}, loss = {1}'.format(i, loss))

        return loss_list

    def predict(self, x_test):
        """
        预测数据。
        :param x_test:
        :return:
        """
        y_pred = x_test.dot(self.w) + self.b
        return y_pred


def least_square_approach(x_array, y_array):
    """
    根据公式计算最小二乘法，输入类型为ndarray。
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
    y_hat = theta * x_array

    common_logger.info("y_hat:\n{0}".format(y_hat))
    common_logger.info(x_array[:, 0].flatten())
    common_logger.info(y_array.T[0, :].flatten())
    # 画图
    plt.figure()
    plt.scatter(x_array[:, 0], y_array.T[0, :])
    plt.plot(x_array[:, 0], y_hat, 'r-')
    plt.show()


def least_square_mat(x_list, y_list):
    """
    使用matrix计算最小二乘法。
    :param x_list: list类型，x输入
    :param y_list: list类型，y输入
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


def local_weight_linear_reg():
    pass


def local_weight_lr_mat(test_point, x_list, y_list, k=1.0):
    """
    使用matrix实现局部加权线性回归。
    w_hat = (xTwx)^(-1)xTwy
    :param test_point:
    :param x_list:
    :param y_list:
    :param k:
    :return:
    """
    x_mat = np.mat(x_list)
    y_mat = np.mat(y_list)

    # shape 读取矩阵的长度 shape[0]获得矩阵第一维的长度
    m = np.shape(x_mat)[0]

    # 创建对角矩阵
    weights = np.mat(np.eye(m))

    # next 2 lines create weights matrix
    for j in range(m):
        # 矩阵每行的差
        diff_mat = test_point - x_mat[j,:]
        # 计算权重
        weights[j, j] = np.exp(diff_mat*diff_mat.T/(-2.0*k**2))
    xTwx = x_mat.T * (weights * x_mat)

    if np.linalg.det(xTwx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTwx.I * (x_mat.T * (weights * y_mat))

    return test_point * ws


def ridge_regression_mat(x_mat, y_mat, lam=0.2):
    """
    岭回归，就是在矩阵xTx上增加一项使得矩阵非奇异，从而能够对其求逆。
    :param x_mat:
    :param y_mat:
    :param lam:
    :return:
    """
    xTx = x_mat.T * x_mat
    denom = xTx + np.eye(np.shape(x_mat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (x_mat.T * y_mat)
    return ws


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
    # least_square_mat(train_x, train_y)
    # 测试sk的线性回归
    # linear_regression_sk(train_x.tolist(), train_y.tolist())
    # 测试本地线性回归
    # linear_reg = LinearRegression()
    # linear_reg.train(train_x, train_y, learn_rate=0.01)
    # common_logger.info("weight:{0}, bias:{1}".format(linear_reg.w, linear_reg.b))
    # linear_reg = LinearRegression(2)
    # linear_reg.train(train_x, train_y, learn_rate=0.001)
    # common_logger.info("weight:{0}, bias:{1}".format(linear_reg.w, linear_reg.b))
    #
    # # 二维求解
    # train_x2 = np.ones(train_x.shape)
    # train_x3 = np.hstack((train_x, train_x2))
    # least_square_approach(train_x3, train_y)
    # least_square_mat(train_x3, train_y)


def test_local_weight_lr():
    """
    测试局部权重线性回归。
    :return:
    """
    train_x = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59,
                          2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
    train_y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53,
                          1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
    train_x = train_x.reshape([-1, 1])
    train_y = train_y.reshape([-1, 1])

    m = np.shape(train_x)[0]
    y_hat = np.zeros(m)
    for i in range(m):
        y_hat[i] = local_weight_lr_mat(train_x[i], train_x, train_y, 1.0)

    # 计算对train_x排序的索引值
    str_ind = train_x[:, 0].argsort(0)
    common_logger.info("str_ind:\n{0}".format(str_ind))
    x_sort = train_x[str_ind]
    common_logger.info("x_sort:\n{0}".format(x_sort))
    common_logger.info("y_hat:\n{0}".format(y_hat))
    # 画图
    plt.figure()
    plt.scatter(train_x[:, 0], train_y.T[0, :])
    plt.plot(x_sort[:, 0], y_hat[str_ind], 'r-')
    plt.show()


def test_ridge_regression():
    """
    测试岭回归。
    :return:
    """
    train_x = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59,
                          2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
    train_y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53,
                          1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])

    x_mat = np.mat(train_x).T
    y_mat = np.mat(train_y).T

    # y_mat 标准化
    y_mean = np.mean(y_mat)
    y_mat -= y_mean

    # x_mat 标准化
    x_means = np.mean(x_mat, 0)
    x_var = np.var(x_mat, 0)
    # （特征-均值）/方差
    x_mat = (x_mat - x_means) / x_var
    # common_logger.info("x_means:\n{0}".format(x_means))
    # common_logger.info("x_var:\n{0}".format(x_var))
    # common_logger.info("x_mat:\n{0}".format(x_mat))
    # common_logger.info("y_mat:\n{0}".format(y_mat))

    num_test_pts = 30
    w_mat = np.zeros((num_test_pts, np.shape(x_mat)[1]))
    for i in range(num_test_pts):  # 测试不同的lambda取值，获得系数
        ws = ridge_regression_mat(x_mat, y_mat, np.exp(i - 10))
        w_mat[i, :] = ws.T

    common_logger.info("w_mat:\n{0}".format(w_mat))

    # plt.plot(w_mat, 'r-')
    # plt.show()


if __name__ == "__main__":
    test_linear_reg()
    # test_local_weight_lr()
    test_ridge_regression()
    pass

