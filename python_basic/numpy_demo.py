# -*- coding:utf-8 -*-

# ==============================================================================
# 测试numpy的相关方法。
# ==============================================================================

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
    common_logger.info("array a:\n{0}".format(a))
    common_logger.info("array a type:\n{0}".format(type(a)))
    common_logger.info("array a shape:\n{0}".format(a.shape))
    common_logger.info("array a ndim:\n{0}".format(a.ndim))
    common_logger.info("array a dtype:\n{0}".format(a.dtype.name))
    # 数组元素在内存中的字节数
    common_logger.info("array a itemsize:\n{0}".format(a.itemsize))

    # matrix是2位数组，已经不再推荐使用。But It is no longer recommended to use this class
    m = np.asmatrix(a)
    m2 = np.matrix([[1, 2], [3, 4]])
    common_logger.info("matrix m:\n{0}".format(m))
    common_logger.info("matrix m type:\n{0}".format(type(m)))
    common_logger.info("matrix m2:\n{0}".format(m2))


def test_ndarray_transform():
    """
    测试ndarray的变形操作。
    :return:
    """
    # ndarray变成list
    nd_a = np.arange(15).reshape(3, 5)
    arr_list = nd_a.tolist()
    common_logger.info("array list:\n{0}".format(arr_list))
    common_logger.info("array type:\n{0}".format(type(arr_list)))

    # 转置操作
    nd_a_t = nd_a.transpose()
    common_logger.info("transpose:\n{0}".format(nd_a_t))
    common_logger.info("na_a.T:\n{0}".format(nd_a.T))


def test_ndarray_operator():
    """
    测试ndarray的运算。
    :return:
    """
    nd_a = np.array([[1, 2], [3, 4]])
    nd_b = np.array([[1], [2]])

    # 矩阵逆运算
    nd_a_inverse = np.linalg.inv(nd_a)
    common_logger.info("inverse:\n{0}".format(nd_a_inverse))
    # 广义逆矩阵 (ATA)-1AT
    nd_a_pinv = np.linalg.pinv(nd_a)
    common_logger.info("pseudo inverse:\n{0}".format(nd_a_pinv))
    # 估计线性模型中的系数
    # coef_a = np.linalg.lstsq(nd_a, nd_y)
    # common_logger.info("nd_a nd_y coef:\n{0}".format(coef_a))
    # 矩阵的行列式
    nd_a_det = np.linalg.det(nd_a)
    common_logger.info("det:\n{0}".format(nd_a_det))
    # 解形如aX=b的线性方程组
    nd_a_solve = np.linalg.solve(nd_a, nd_b)
    common_logger.info("solve:\n{0}".format(nd_a_solve))
    # 矩阵的特征值
    nd_a_eigvals = np.linalg.eigvals(nd_a)
    common_logger.info("eigval:\n{0}".format(nd_a_eigvals))
    # 特征值和特征向量
    nd_a_eig = np.linalg.eig(nd_a)
    common_logger.info("eig:\n{0}".format(nd_a_eig))
    # svd分解
    nd_a_svd = np.linalg.svd(nd_a)
    common_logger.info("svd:\n{0}".format(nd_a_svd))


def test_stack():
    """
    ndarray叠加，重组。
    :return:
    """
    a = np.floor(10 * np.random.random((2,2)))
    b = np.floor(10 * np.random.random((2, 2)))
    v_stack = np.vstack((a, b))
    h_stack = np.hstack((a, b))
    common_logger.info("random a:\n{0}".format(a))
    common_logger.info("random b:\n{0}".format(b))
    common_logger.info("v_stack:\n{0}".format(v_stack))
    common_logger.info("h_stack:\n{0}".format(h_stack))


if __name__ == "__main__":
    # test_create_ndarray()
    # test_ndarray_attribute()
    # test_ndarray_transform()
    # test_ndarray_operator()
    test_stack()
    pass
