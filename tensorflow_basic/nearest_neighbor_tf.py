# -*- coding:utf-8 -*-

# ==============================================================================
# tensorflow版本的线性回归模型。
# ==============================================================================
import tensorflow as tf
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import config.common_config as com_config
from util.log_util import LoggerUtil


# 日志器
common_logger = LoggerUtil.get_common_logger()
# mnist数据
mnist = input_data.read_data_sets(com_config.MNIST_DIR, one_hot=True)


class NearestNeighborTF(object):
    """
    最近邻模型TensorFlow版本。
    """
    def __init__(self, x_data, y_data, input_dim):
        """
        初始化线性回归模型。
        :param x_data: 输入x数据
        :param y_data: 输入y数据
        """
        # 网络结构参数
        self.train_input = None
        self.test_input = None
        self.input_dim = input_dim
        self.x_data = x_data
        self.y_data = y_data

        # 预测优化函数
        self.pred_index = None

        # sess会话
        self.sess = None

        # 初始化器
        self.initializer = None
        self.merged_summary_op = None

        # 目录初始化
        self.tf_logs_path = os.path.join(com_config.TF_MODEL_LOGS_DIR, "NearestNeighborTF")
        self.checkpoints_path = os.path.join(com_config.TF_MODEL_CHECKPOINT_DIR, "NearestNeighborTF")

        # 变量和参数初始化
        self.init_network()

    @staticmethod
    def name(self):
        """
        结构名称。
        :return:
        """
        return "Nearest Neighbor TensorFlow(NN)"

    def init_network(self):
        """
        初始化网络结构信息。
        :return:
        """
        self.placeholders()
        self.inference()
        # 初始化变量的操作应该放在最后
        self.init_variables()

    def placeholders(self):
        """
        使用到的占位符。
        :return:
        """
        self.train_input = tf.placeholder(tf.float32, shape=[None, self.input_dim], name="train_input")
        self.test_input = tf.placeholder(tf.float32, shape=[self.input_dim], name="test_input")

    def init_variables(self):
        """
        初始化变量。
        :return:
        """
        self.initializer = tf.global_variables_initializer()

    def inference(self):
        """
        网络结构生成。
        :return:
        """
        # 计算预测值
        distance = tf.reduce_sum(tf.abs(tf.add(self.train_input, tf.negative(self.test_input))), axis=1)
        self.pred_index = tf.argmin(distance, 0)

    def train_mnist(self):
        """
        训练mnist网络参数。
        :return:
        """
        with tf.Session() as sess:
            self.sess = sess
            # 执行初始化
            sess.run(self.initializer)

            test_x, test_y = mnist.test.images, mnist.test.labels
            accuracy = 0.

            # 循环运行所有测试数据
            for i in range(len(test_x)):
                nn_index = sess.run(self.pred_index, feed_dict={self.train_input: self.x_data, self.test_input: test_x[i, :]})
                predict_value = np.argmax(self.y_data[nn_index])
                label_value = np.argmax(test_y[i])

                if predict_value == label_value:
                    accuracy += 1. / len(test_x)
                else:
                    common_logger.info("Test {0} Prediction: {1} Actual Class:{2}".format(i, predict_value, label_value))
            common_logger.info("Accuracy:{0}".format(accuracy))


def nearest_neighbor_model():
    """
    最近邻模型。
    :return:
    """
    train_x, train_y = mnist.train.images, mnist.train.labels
    test_x, test_y = mnist.test.images, mnist.test.labels

    # tf Graph Input
    train_input = tf.placeholder("float", [None, 784])
    test_input = tf.placeholder("float", [784])

    # Nearest Neighbor calculation using L1 Distance
    # Calculate L1 Distance
    distance = tf.reduce_sum(tf.abs(tf.add(train_input, tf.negative(test_input))), axis=1)
    pred_index = tf.argmin(distance, 0)

    accuracy = 0.

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(len(test_x)):
            # Get nearest neighbor
            nn_index = sess.run(pred_index, feed_dict={train_input: train_x, test_input: test_x[i, :]})
            predict_value = np.argmax(train_y[nn_index])
            label_value = np.argmax(test_y[i])

            if predict_value == label_value:
                accuracy += 1. / len(test_x)
            else:
                common_logger.info("Test {0} Prediction: {1} Actual Class:{2}".format(i, predict_value, label_value))

        common_logger.info("Accuracy:{0}".format(accuracy))


def test_train():
    """
    测试训练数据的过程。
    :return:
    """
    # Training Data
    train_x, train_y = mnist.train.images, mnist.train.labels
    nn_tf = NearestNeighborTF(train_x, train_y, train_x.shape[1])
    nn_tf.train_mnist()


if __name__ == "__main__":
    # nearest_neighbor_model()
    test_train()
    pass
