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
    测试tf.concat连接操作。tf.concat不会更改tensor的维度。
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
    test_run_sess("连接操作 axis=0 生成结果的 shape：", tf_shape1)
    test_run_sess("连接操作 axis=1 生成结果的 shape：", tf_shape2)


def test_run_sess(desc, tf_op):
    """
    测试在sess中运行tf操作。
    :param desc: 操作描述
    :param tf_op:
    :return:
    """
    with tf.Session() as sess:
        result = sess.run(tf_op)
        common_logger.info("{0}:\n{1}".format(desc, result))


def test_nn_conv2d():
    """
    测试nn.conv2d二位卷积操作。
    :return:
    """
    # 构造输入数据
    v_list = [i for i in range(25)]
    v_const = tf.constant(v_list)
    v_const = tf.cast(v_const, "float32")
    v_image = tf.reshape(v_const, shape=[-1, 5, 5, 1])

    # 构造权重和卷积核
    v_weight = tf.constant([1, 1, 1, 1])
    v_weight = tf.cast(v_weight, "float32")
    w_filter = tf.reshape(v_weight, shape=[2, 2, 1, 1])

    # 卷积运算
    v_conv1 = tf.nn.conv2d(v_image, filter=w_filter, strides=[1, 1, 1, 1], padding="VALID")

    with tf.Session() as sess:
        kernel = sess.run(w_filter)
        result_shape = sess.run(tf.shape(v_conv1))
        result_conv = sess.run(v_conv1)
        common_logger.info("kernel:{0}".format(kernel))
        common_logger.info("shape:{0}".format(result_shape))
        common_logger.info("value:{0}".format(result_conv))

    # 修改权重和卷积核
    v_shape = [2, 2, 1, 2]
    v_weight = tf.truncated_normal(v_shape, mean=1, stddev=0)
    v_weight = tf.cast(v_weight, "float32")

    # 卷积运算
    v_conv2 = tf.nn.conv2d(v_image, filter=v_weight, strides=[1, 1, 1, 1], padding="VALID")

    with tf.Session() as sess:
        kernel = sess.run(v_weight)
        result_shape = sess.run(tf.shape(v_conv2))
        result_conv = sess.run(v_conv2)
        common_logger.info("kernel:{0}".format(kernel))
        common_logger.info("shape:{0}".format(result_shape))
        common_logger.info("value:{0}".format(result_conv))


def test_expand_dims():
    """
    测试tf.expand_dims方法。扩充维度。
    :return:
    """
    t1 = tf.constant([1, 2, 3])
    t2 = tf.constant([4, 5, 6])
    # concated = tf.concat(1, [t1,t2])这样会报错，tf.concat不会更改tensor的维度。
    t1 = tf.expand_dims(tf.constant([1, 2, 3]), 1)
    t2 = tf.expand_dims(tf.constant([4, 5, 6]), 1)
    t_concate = tf.concat([t1, t2], 1)

    test_run_sess("expand_dim t1", t1)
    test_run_sess("expand_dim t2", t2)
    test_run_sess("concated", t_concate)


def test_stack():
    """
    测试tf.stack方法。tf.pack已经修改维tf.stack方法。
    :return:
    """
    x = tf.constant([1, 4])
    y = tf.constant([2, 5])
    z = tf.constant([3, 6])
    t_stack0 = tf.stack([x, y, z])  # [[1, 4], [2, 5], [3, 6]] (Pack along first dim.)
    t_stack1 = tf.stack([x, y, z], axis=1)  # [[1, 2, 3], [4, 5, 6]]
    t_stack2 = tf.stack([5, 4])

    test_run_sess("stack 0", t_stack0)
    test_run_sess("stack 1", t_stack1)
    test_run_sess("stack 2", t_stack2)


def test_spark_to_dense():
    """
    测试稀疏矩阵到稠密矩阵的转换tf.sparse_to_dense。
    :return:
    """
    batch_size = 6
    label = tf.expand_dims(tf.constant([0, 2, 3, 6, 7, 9]), 1)
    index = tf.expand_dims(tf.range(0, batch_size), 1)

    # use a matrix
    concated1 = tf.concat([index, label], 1)
    onehot_labels1 = tf.sparse_to_dense(concated1, tf.stack([batch_size, 10]), 1.0, 0.0)

    # use a vector
    concated2 = tf.constant([1, 3, 4])
    onehot_labels2 = tf.sparse_to_dense(concated2, tf.stack([10]), 1.0, 0.0)  # can use
    onehot_labels4 = tf.sparse_to_dense(concated2, tf.stack([10]), concated2, 0)

    # use a scalar
    concated3 = tf.constant(5)
    onehot_labels3 = tf.sparse_to_dense(concated3, tf.stack([10]), 1.0, 0.0)

    test_run_sess("onehot_labels 1", onehot_labels1)
    test_run_sess("onehot_labels 2", onehot_labels2)
    test_run_sess("onehot_labels 3", onehot_labels3)
    test_run_sess("onehot_labels 4", onehot_labels4)


if __name__ == "__main__":
    # test_reshape()
    # test_transpose()
    # test_truncate_norm()
    # test_concat()
    # test_nn_conv2d()
    # test_expand_dims()
    # test_stack()
    test_spark_to_dense()
    pass

