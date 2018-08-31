# -*- coding:utf-8 -*-

# ==============================================================================
# tensorflow常用操作的工具类。
# ==============================================================================
import tensorflow as tf
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()


class TensorFlowUtil(object):
    @staticmethod
    def print_activations(t):
        """
        显示网络每一层结构，展示每一个卷积层或池化层输出tensor的尺寸
        :return:
        """
        common_logger.info("{0}{1}".format(t.op.name, t.get_shape().as_list()))

    @staticmethod
    def weight_variable(shape, mode="truncated_normal", name="weight"):
        """
        构造权重变量。
        :param shape:
        :param mode:权重的模式
        :param name:名称
        :return:
        """
        if mode == "truncated_normal":
            initial = tf.truncated_normal(shape, stddev=0.1)
        elif mode == "random_normal":
            initial = tf.random_normal(shape, stddev=0.1)
        elif mode == "constant":
            initial = tf.constant(0.1, shape=shape)
        else:
            initial = tf.truncated_normal(shape, stddev=0.1)

        return tf.Variable(initial, name=name)

    @staticmethod
    def bias_variable(shape, mode="constant", name="bias"):
        """
        构造偏置变量。
        :param shape:
        :param mode:权重的模式
        :param name:名称
        :return:
        """
        if mode == "constant":
            initial = tf.constant(0.1, shape=shape)
        elif mode == "zeros":
            initial = tf.zeros(shape=shape)
        else:
            initial = tf.zeros(shape=shape)

        return tf.Variable(initial, name=name)

    @staticmethod
    def conv2d(x, w, b, strides=1):
        """
        二维卷积操作。
        :param x: 输入tensor
        :param w: 权重变量
        :param b: 偏置变量
        :param strides: 步长
        :return:
        """
        result = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
        result = tf.nn.bias_add(result, b)
        return result

    @staticmethod
    def maxpool2d(x, k=2, strides=2):
        """
        最大池化操作。
        :param x: 输入tensor
        :param k: 池化核ksize
        :param strides: 步长
        :return:
        """
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, strides, strides, 1], padding='SAME')

    @staticmethod
    def relu(x):
        """
        relu操作。
        :param x:
        :return:
        """
        return tf.nn.relu(x)

    @staticmethod
    def conv_relu_maxpool(x, w, b, strides=1, k=2):
        """
        2维卷积整流最大池化层。
        :param x: 输入
        :param w: 卷积核
        :param b: 偏置
        :param strides: 卷积步长
        :param k: 最大池化核大小
        :return:
        """
        layer_conv = TensorFlowUtil.conv2d(x, w, b, strides=strides)
        layer_relu = TensorFlowUtil.relu(layer_conv)
        layer_conv_pool = TensorFlowUtil.maxpool2d(layer_relu, k)
        return layer_conv_pool

    @staticmethod
    def conv_relu(x, w, b, strides=1):
        """
        2维卷积整流层。
        :param x: 输入
        :param w: 卷积核
        :param b: 偏置
        :param strides: 卷积步长
        :return:
        """
        layer_conv = TensorFlowUtil.conv2d(x, w, b, strides=strides)
        layer_conv_relu = TensorFlowUtil.relu(layer_conv)
        return layer_conv_relu

    @staticmethod
    def fc(x, w, b):
        """
        全连接操作, w*x+b。
        :param x: 输入
        :param w: 权重
        :param b: 偏置
        :return:
        """
        return tf.add(tf.matmul(x, w), b)

    @staticmethod
    def fc_relu(x, w, b):
        """
        全连接操作以及relu激活, w*x+b。
        :param x: 输入
        :param w: 权重
        :param b: 偏置
        :return:
        """
        return TensorFlowUtil.relu(tf.add(tf.matmul(x, w), b))
