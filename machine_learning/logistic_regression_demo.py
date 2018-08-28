# -*- coding:utf-8 -*-

# ==============================================================================
# 测试逻辑回归的相关方法。
# ==============================================================================
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from util.log_util import LoggerUtil
import config.common_config as com_config

# 日志器
common_logger = LoggerUtil.get_common_logger()
resource_dir = com_config.RESOURCE_DIR
ml_data_dir = os.path.join(resource_dir, "ml_data")


def sigmoid(in_x):
    """
    计算sigmoid的函数。
    :param in_x: ndarray类型的输入数据。
    :return:
    """
    return 1.0 / (1 + np.exp(-in_x))


def load_data_set():
    """
    加载数据集。
    :return:
    """
    data_mat = []
    label_mat = []

    file_path = os.path.join(ml_data_dir, "logistic_test_set.txt")
    with open(file_path) as file:
        for line in file:
            line_arr = line.strip().split()
            data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])  # x0=1
            label_mat.append(int(line_arr[2]))

    return data_mat, label_mat


def grad_descent(x_inputs, y_labels):
    """
    使用梯度下降计算theta。
    :param x_inputs:
    :param y_labels:
    :return:
    """
    x_input_mat = np.mat(x_inputs)
    y_label_mat = np.mat(y_labels).transpose()

    # 学习率
    alpha = 0.001
    # 最大迭代次数
    max_iters = 500

    n_datas, n_features = np.shape(x_input_mat)
    # 初始化theta
    theta = np.ones((n_features, 1))

    for k in range(max_iters):
        gradient = x_input_mat * theta
        y_predict = sigmoid(gradient)
        error = (y_predict - y_label_mat)
        theta -= alpha * x_input_mat.transpose() * error
    return theta


def stochastic_grad_descent(x_inputs, y_labels):
    """
    使用随机梯度下降计算theta。
    :param x_inputs:
    :param y_labels:
    :return:
    """
    x_input_mat = np.mat(x_inputs)
    y_label_mat = np.mat(y_labels).transpose()

    # 学习率
    alpha = 0.001
    # 最大迭代次数
    max_iters = 500

    n_datas, n_features = np.shape(x_input_mat)
    # 初始化theta
    theta = np.ones((n_features, 1))

    # 逐条输入数据进行迭代
    for k in range(max_iters):
        for idx in range(n_datas):
            gradient = x_input_mat[idx] * theta
            y_predict = sigmoid(gradient)
            error = (y_predict - y_label_mat[idx])
            theta -= alpha * x_input_mat[idx].transpose() * error
    return theta


def smooth_stoc_grad_descent(x_inputs, y_labels):
    """
    使用平滑的随机梯度下降计算theta。
    :param x_inputs:
    :param y_labels:
    :return:
    """
    x_input_mat = np.mat(x_inputs)
    y_label_mat = np.mat(y_labels).transpose()

    # 最大迭代次数
    max_iters = 500

    n_datas, n_features = np.shape(x_input_mat)
    # 初始化theta
    theta = np.ones((n_features, 1))

    # 逐条输入数据进行迭代
    for k in range(max_iters):
        data_index = list(range(n_datas))
        for i in range(n_datas):
            # alpha会随着迭代次数不断减小，但存在常数项，它不会小到0
            # 这种设置可以缓解数据波动
            alpha = 4/(1.0 + k + i) + 0.001
            # 通过随机选取样本来更新回归系数
            rand_index = int(np.random.uniform(0, len(data_index)))
            x_sum = np.sum(x_input_mat[rand_index] * theta)
            y_predict = sigmoid(x_sum)
            error = y_predict - y_label_mat[rand_index]
            theta -= alpha * x_input_mat[rand_index].transpose() * error
            del(data_index[rand_index])
    return theta


def logistic_regression_sk(x, y):
    """
    使用sklearn的LogisticRegression。
    :param x:
    :param y:
    :return:
    """
    logistic_estimator = linear_model.LogisticRegression()
    logistic_estimator.fit(x, y)

    common_logger.info("coef:{0}, intercept:{1}".format(logistic_estimator.coef_, logistic_estimator.intercept_))
    # 构造theta
    theta = logistic_estimator.coef_[0]
    theta[0] += logistic_estimator.intercept_
    return theta


def predict_classify(in_x, theta):
    """
    预测分类函数。
    :param in_x: 输入数据
    :param theta:
    :return:
    """
    prob = sigmoid(np.sum(in_x*theta))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def test_logistic_regression():
    """
    测试逻辑回归算法。
    :return:
    """
    x_inputs, y_labels = load_data_set()
    # theta = grad_descent(x_inputs, y_labels)
    # common_logger.info("theta:\n{0}".format(theta))

    # theta_stoch = stochastic_grad_descent(x_inputs, y_labels)
    # common_logger.info("theta_stoch:\n{0}".format(theta_stoch))

    theta_smooth_stoch = smooth_stoc_grad_descent(x_inputs, y_labels)
    common_logger.info("theta_smooth_stoch:\n{0}".format(theta_smooth_stoch))

    # theta_sk = logistic_regression_sk(x_inputs, y_labels)

    theta_list = [theta_smooth_stoch]
    # 显示绘图
    plot_best_fit(theta_list)


def plot_best_fit(theta_list):
    """
    根据训练好的theta绘图。
    :param theta_list:
    :return:
    """
    # 加载数据
    data_mat, label_mat = load_data_set()
    data_arr = np.array(data_mat)
    n_datas = np.shape(data_arr)[0]

    # x, y坐标
    x_cord1 = []
    y_cord1 = []
    x_cord2 = []
    y_cord2 = []

    # 将数据按真实标签进行分类
    for i in range(n_datas):
        if int(label_mat[i]) == 1:
            x_cord1.append(data_arr[i, 1])
            y_cord1.append(data_arr[i, 2])
        else:
            x_cord2.append(data_arr[i, 1])
            y_cord2.append(data_arr[i, 2])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_cord1, y_cord1, s=30, c='red', marker='s')
    ax.scatter(x_cord2, y_cord2, s=30, c='blue')
    # 生成x的取值-3.0--3.0, 增量为0.1
    x1 = np.arange(-3.0, 3.0, 0.1)
    # 根据y=Θ0+Θ1x1+Θ2x2, 求y=0时的x2
    for theta in theta_list:
        x2 = (-theta[0] - theta[1] * x1) / theta[2]
        ax.plot(x1, x2.T)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


if __name__ == "__main__":
    test_logistic_regression()
    pass

