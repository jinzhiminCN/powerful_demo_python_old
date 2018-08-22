# -*- coding:utf-8 -*-

# ==============================================================================
# tensorflow基本操作测试。
# ==============================================================================
import tensorflow as tf
import os
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()


def test_reshape():
    """
    测试tf.reshape运算。
    :return:
    """
    values = [x for x in range(0, 12)]

    tf_const1 = tf.constant(values)
    tf_reshape1 = tf.reshape(tf_const1, shape=(3, 4, 1))
    tf_reshape2 = tf.reshape(tf_reshape1, shape=(4, 3, 1))

    with tf.Session() as sess:
        val1, val2, val3 = sess.run([tf_const1, tf_reshape1, tf_reshape2])
        common_logger.info("原始数据:{0}".format(val1))
        common_logger.info("变形(3*4)后:{0}".format(val2))
        common_logger.info("变形(4*3)后:{0}".format(val3))

    # 一维变二维
    values = [x for x in range(1, 10)]
    tf_const1 = tf.constant(values)
    tf_reshape1 = tf.reshape(tf_const1, [3, 3])
    test_run_sess("一维变二维：", tf_reshape1)

    values = [[x, x, x] for x in range(1, 7)]
    tf_const = tf.constant(values)
    tf_reshape = tf.reshape(tf_const, [3, 2, 3])
    tf_reshape1 = tf.reshape(tf_reshape, [-1])
    tf_reshape2 = tf.reshape(tf_reshape, [2, -1])
    tf_reshape3 = tf.reshape(tf_reshape, [-1, 9])
    tf_reshape4 = tf.reshape(tf_reshape, [3, -1, 2])

    test_run_sess("原始数据：", tf_reshape)
    test_run_sess("一维平铺[-1]：", tf_reshape1)
    test_run_sess("变形推断[2, -1]：", tf_reshape2)
    test_run_sess("变形推断[-1, 9]：", tf_reshape3)
    test_run_sess("变形推断[3, -1, 2]：", tf_reshape4)

    # shape `[]` reshapes to a scalar
    tf_reshape = tf.reshape([[1]], [])
    test_run_sess("标量数据：", tf_reshape)


def test_transpose():
    """
    测试tf.transpose转置操作。
    :return:
    """
    # 默认转置
    x_const = tf.constant([[1, 2, 3], [4, 5, 6]])
    tf_transpose1 = tf.transpose(x_const)
    tf_transpose2 = tf.transpose(x_const, perm=[1, 0])
    test_run_sess("原始数据：", x_const)
    test_run_sess("默认转置：", tf_transpose1)
    test_run_sess("定向转置：", tf_transpose2)

    # 矩阵转置
    x = tf.constant([[[1, 2, 3],
                      [4, 5, 6]],
                     [[7, 8, 9],
                      [10, 11, 12]]])
    tf_transpose3 = tf.transpose(x, perm=[0, 2, 1])
    tf_transpose4 = tf.transpose(x, perm=[2, 0, 1])
    test_run_sess("定向转置[0, 2, 1]：", tf_transpose3)
    test_run_sess("定向转置[2, 0, 1]：", tf_transpose4)


def test_truncate_norm():
    """
    测试截尾正态分布tf.truncated_normal操作。
    :return:
    """
    norm_result = tf.truncated_normal([10], stddev=0.1)
    test_run_sess("截尾正态分布随机值：", norm_result)

    sum_result = tf.reduce_sum(norm_result)
    test_run_sess("截尾正态分布随机值求和：", sum_result)

    norm_result = tf.truncated_normal([10, 3], stddev=0.1)
    test_run_sess("截尾正态分布2维随机值：", norm_result)


def test_concat():
    """
    测试tf.concat连接操作。
    :return:
    """
    t1 = [[1, 2, 3], [4, 5, 6]]
    t2 = [[7, 8, 9], [10, 11, 12]]
    tf_concat1 = tf.concat([t1, t2], 0)
    tf_concat2 = tf.concat([t1, t2], 1)
    tf_shape1 = tf.shape(tf_concat1)
    tf_shape2 = tf.shape(tf_concat2)
    test_run_sess("连接操作 axis=0：", tf_concat1)
    test_run_sess("连接操作 axis=1：", tf_concat2)
    test_run_sess("连接操作 axis=0 shape：", tf_shape1)
    test_run_sess("连接操作 axis=1 shape：", tf_shape2)


def test_run_sess(desc, tf_op):
    """
    测试在sess中运行tf操作。
    :param desc: 操作描述
    :param tf_op:
    :return:
    """
    with tf.Session() as sess:
        result = sess.run(tf_op)
        common_logger.info("{0}:{1}".format(desc, result))


if __name__ == "__main__":
    # test_reshape()
    # test_transpose()
    # test_truncate_norm()
    test_concat()
    pass

