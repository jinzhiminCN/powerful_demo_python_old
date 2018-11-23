# -*- coding:utf-8 -*-

# ==============================================================================
# 测试numpy的相关方法。
# ==============================================================================
import scipy.stats as stats
import numpy as np
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()


def test_create_ndarray():
    """
    测试创建ndarray。
    :return:
    """
    nd_zeros = np.zeros([3, 5])
    nd_ones = np.ones([3, 5])
    nd_rand = np.random.rand(3, 4)
    nd_eye = np.eye(3)
    nd_eye1 = np.eye(3, k=1)

    common_logger.info(nd_zeros)
    common_logger.info(nd_ones)
    common_logger.info(nd_rand)
    common_logger.info(nd_eye)
    common_logger.info(nd_eye1)


def test_ndarray_attribute():
    """
    测试ndarray的属性。
    :return:
    """
    # ndarray是n维数组
    a = np.arange(15).reshape(3, 5)
    log_print_value("array a", a)
    log_print_value("array a type", type(a))
    log_print_value("array a shape", a.shape)
    log_print_value("array a ndim", a.ndim)
    log_print_value("array a dtype", a.dtype.name)
    # 数组元素在内存中的字节数
    log_print_value("array a itemsize", a.itemsize)

    # matrix是2位数组，已经不再推荐使用。But It is no longer recommended to use this class
    m = np.asmatrix(a)
    m2 = np.matrix([[1, 2], [3, 4]])
    m3 = np.mat([[10, 20], [30, 40]])

    log_print_value("matrix m", m)
    log_print_value("matrix m type", type(m))
    log_print_value("matrix m2", m2)
    log_print_value("matrix m3", m3)


def test_ndarray_transform():
    """
    测试ndarray的变形操作。
    :return:
    """
    # ndarray变成list
    nd_a = np.arange(15).reshape(3, 5)
    arr_list = nd_a.tolist()
    log_print_value("array list", arr_list)
    log_print_value("array type", type(arr_list))

    # 转置操作
    nd_a_t = nd_a.transpose()
    log_print_value("transpose", nd_a_t)
    log_print_value("na_a.T", nd_a.T)


def test_ndarray_operator():
    """
    测试ndarray的运算。
    :return:
    """
    nd_a = np.array([[1, 2], [3, 4]])
    nd_b = np.array([[1], [2]])

    # 矩阵逆运算
    nd_a_inverse = np.linalg.inv(nd_a)
    log_print_value("inverse", nd_a_inverse)

    # 广义逆矩阵 (ATA)-1AT
    nd_a_pinv = np.linalg.pinv(nd_a)
    log_print_value("pseudo inverse", nd_a_pinv)

    # 估计线性模型中的系数
    # coef_a = np.linalg.lstsq(nd_a, nd_y)
    # log_print_value("nd_a nd_y coef", coef_a))

    # 矩阵的行列式
    nd_a_det = np.linalg.det(nd_a)
    log_print_value("det", nd_a_det)

    # 解形如aX=b的线性方程组
    nd_a_solve = np.linalg.solve(nd_a, nd_b)
    log_print_value("solve", nd_a_solve)

    # 矩阵的特征值
    nd_a_eigvals = np.linalg.eigvals(nd_a)
    log_print_value("eigval", nd_a_eigvals)

    # 特征值和特征向量
    nd_a_eig = np.linalg.eig(nd_a)
    log_print_value("eig", nd_a_eig)

    # svd分解
    nd_a_svd = np.linalg.svd(nd_a)
    log_print_value("svd", nd_a_svd)


def test_stack():
    """
    ndarray堆叠数组。
    :return:
    """
    a = np.floor(10 * np.random.random((2, 2)))
    b = np.floor(10 * np.random.random((2, 2)))
    v_stack = np.vstack((a, b))
    h_stack = np.hstack((a, b))

    log_print_value("random a", a)
    log_print_value("random b", b)
    log_print_value("v_stack", v_stack)
    log_print_value("h_stack", h_stack)


def log_print_value(np_name, np_value):
    """
    使用日志打印numpy的运算结果。
    :param np_name
    :param np_value
    :return:
    """
    common_logger.info("{0}:\n{1}".format(np_name, np_value))


def test_reshape():
    """
    测试reshape操作。
    :return:
    """
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    groups = 3
    channels_per_group = int(len(x)/groups)

    x_reshape = x.reshape(groups, channels_per_group)
    x_reshape = x_reshape.T
    x_reshape = x_reshape.reshape(groups * channels_per_group)
    log_print_value("reshape", x_reshape)

    x_reshape = x.reshape(2, 2, 3)
    log_print_value("reshape", x_reshape)


def test_statistic():
    """
    测试统计量。
    :return:
    """
    data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 10, 11])
    # 计算均值
    data_mean = np.mean(data)
    # 计算中位数
    data_median = np.median(data)
    # 计算众数
    data_mode = stats.mode(data)

    log_print_value("mean", data_mean)
    log_print_value("median", data_median)
    log_print_value("mode", data_mode)

    # 极差 最大值和最小值的发散程度指标
    data_ptp = np.ptp(data)
    # 方差
    data_var = np.var(data)
    # 标准差
    data_std = np.std(data)
    # 变异系数 标准差的无量纲处理
    data_var_coef = np.std(data) / np.mean(data)

    log_print_value("ptp", data_ptp)
    log_print_value("var", data_var)
    log_print_value("std", data_std)
    log_print_value("var_coef", data_var_coef)

    # 计算z-score
    for value in data:
        z_score = get_z_score(value, data_mean, data_std)
        common_logger.info(z_score)

    data2 = np.array([0, 1, 3, 2, 6, 5, 4, 7, 9, 7, 8, 10, 12])
    data_list = np.array([data, data2])

    # 计算两组数的协方差
    data_cov_bias = np.cov(data_list, bias=True)
    data_cov = np.cov(data_list, bias=False)
    data_corrcoef = np.corrcoef(data_list)

    log_print_value("cov_bias", data_cov_bias)
    log_print_value("cov", data_cov)
    log_print_value("corrcoef", data_corrcoef)


def get_z_score(value, data_mean, data_std):
    """
    Z-Score，测量值距均值相差的标准差数目。通常来说，z-分数的绝对值大于3将视为异常。
    :return:
    """
    return (value - data_mean) / data_std


if __name__ == "__main__":
    # test_create_ndarray()
    # test_ndarray_attribute()
    # test_ndarray_transform()
    # test_ndarray_operator()
    # test_stack()
    # test_arg_function()
    # test_reshape()
    test_statistic()
    pass
