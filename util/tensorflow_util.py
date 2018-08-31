# -*- coding:utf-8 -*-

# ==============================================================================
# tensorflow常用操作的工具类。
# ==============================================================================
import tensorflow as tf


class TensorFlowUtil(object):
    @staticmethod
    def weight_variable(shape, mode="truncated_normal"):
        """
        构造权重变量。
        :param shape:
        :param mode:权重的模式
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

        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape, mode="constant"):
        """
        构造偏置变量。
        :param shape:
        :param mode:权重的模式
        :return:
        """
        if mode == "constant":
            initial = tf.constant(0.1, shape=shape)
        else:
            initial = tf.zeros(shape=shape)

        return tf.Variable(initial)

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
    def maxpool2d(x, k=2):
        """
        最大池化操作。
        :param x: 输入tensor
        :param k: ksize
        :return:
        """
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    @staticmethod
    def relu(x):
        """
        relu操作。
        :param x:
        :return:
        """
        return tf.nn.relu(x)
