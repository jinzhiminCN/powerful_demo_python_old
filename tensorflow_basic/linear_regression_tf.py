# -*- coding:utf-8 -*-

# ==============================================================================
# tensorflow版本的线性回归模型。
# ==============================================================================
import tensorflow as tf
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import config.common_config as com_config
from util.log_util import LoggerUtil


# 日志器
common_logger = LoggerUtil.get_common_logger()


class LinearRegressionTF(object):
    """
    线性回归模型TensorFlow版本。
    """
    def __init__(self, x_data, y_data, input_dim):
        """
        初始化线性函数。
        :param x_data: 输入x数据
        :param y_data: 输入y数据
        """
        # 网络结构参数
        self.x_input = None
        self.y_input = None
        self.input_dim = input_dim
        self.weight = None
        self.bias = None
        self.x_data = x_data
        self.y_data = y_data

        # 训练需要的超参数
        self.learning_rate = 1e-3
        self.training_epochs = 2000
        self.batch_size = 1
        self.display_steps = 50
        self.save_steps = 100

        # 预测优化函数
        self.y_value = None
        self.loss = None
        self.optimizer = None

        # sess会话
        self.sess = None

        # 初始化器
        self.initializer = None
        self.merged_summary_op = None

        # 目录初始化
        self.tf_logs_path = os.path.join(com_config.TF_MODEL_LOGS_DIR, "LinearRegressionTF")
        self.checkpoints_path = os.path.join(com_config.TF_MODEL_CHECKPOINT_DIR, "LinearRegressionTF")

        # 变量和参数初始化
        self.init_network()

    @staticmethod
    def name(self):
        """
        结构名称。
        :return:
        """
        return "Linear Regression TensorFlow(LR)"

    def init_network(self):
        """
        初始化网络结构信息。
        :return:
        """
        self.placeholders()
        self.variables()
        self.inference()
        self.loss_function()
        self.evaluate_function()
        self.summary()
        self.solver()
        # 初始化变量的操作应该放在最后
        self.init_variables()

    def prepare_data(self):
        """
        准备数据。
        :return:
        """
        pass

    def placeholders(self):
        """
        使用到的占位符。
        :return:
        """
        # 模型的输入x值
        self.x_input = tf.placeholder(tf.float32, shape=[None, self.input_dim], name="x_input")
        # 模型的输入y值
        self.y_input = tf.placeholder(tf.float32, name="y_input")

    def variables(self):
        """
        使用到的变量。
        :return:
        """
        # 权重变量
        # 1. 权重初始化为0
        # self.weight = tf.Variable(tf.zeros([self.input_dim, 1]), name='Weights')
        # 2. 权重初始化为随机值
        self.weight = tf.Variable(tf.random_uniform([self.input_dim, 1], -1.0, 1.0), name='Weights')

        # 偏置变量
        # 1. 偏置初始化为0
        self.bias = tf.Variable(tf.zeros([1]), name='Bias')
        # 2. 偏置初始化为随机值
        # self.bias = tf.Variable(np.random.randn(), name='Bias')

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
        self.y_value = tf.matmul(self.x_input, self.weight) + self.bias

    def loss_function(self):
        """
        损失函数设置。
        :return:
        """
        n_samples = len(self.x_data)
        # self.loss = tf.reduce_sum(tf.pow(self.y_input - self.y_value, 2)) / (2 * n_samples)
        # 1. 差平方求和除以2倍样本数，配合学习率1e-2
        # self.loss = tf.reduce_mean(tf.square(self.y_input - self.y_value)) / (2 * n_samples)
        # 2. 差平方求和，配合学习率1e-3
        self.loss = tf.reduce_mean(tf.square(self.y_input - self.y_value))

    def evaluate_function(self):
        """
        评价函数设置。
        :return:
        """
        pass

    def solver(self):
        """
        求解器。
        :return:
        """
        # 1. Gradient Descent
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        # 2. Adam
        # self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def summary(self):
        """
        summary 网络结构运行状况概要。
        :return:
        """
        tf.summary.scalar("loss", self.loss)
        self.merged_summary_op = tf.summary.merge_all()

    def train(self):
        """
        训练网络参数。
        :return:
        """
        with tf.Session() as sess:
            self.sess = sess
            # 执行初始化
            sess.run(self.initializer)
            # summary writer
            summary_writer = tf.summary.FileWriter(self.tf_logs_path,
                                                   graph=tf.get_default_graph())
            # saver
            saver = tf.train.Saver()

            # 训练数据
            for epoch in range(self.training_epochs):
                avg_cost = 0.
                total_size = len(self.x_data)
                total_batch = math.ceil(total_size / self.batch_size)

                # 循环运行所有批次数据
                for i in range(total_batch):
                    limit_begin = i * self.batch_size
                    limit_end = max(min((i + 1) * self.batch_size, total_size), 0)
                    batch_xs = self.x_data[limit_begin:limit_end]
                    batch_ys = self.y_data[limit_begin:limit_end]

                    # 执行优化、损失函数
                    _, cost_value, summary = \
                        sess.run([self.optimizer, self.loss, self.merged_summary_op],
                                 feed_dict={self.x_input: batch_xs, self.y_input: batch_ys})

                    # 在summary_writer中记录相应的训练过程
                    summary_writer.add_summary(summary, epoch * total_batch + i)
                    # 计算平均损失
                    avg_cost += cost_value / total_batch

                    # common_logger.info("Epoch: {0:0>4}_{1:0>4} cost={2:.9f}".format(
                    #     (epoch + 1), i, cost_value))

                # 记录每轮迭代的中间结果
                if (epoch + 1) % self.display_steps == 0:
                    common_logger.info("Epoch: {0:0>4} cost={1:.9f}".format((epoch + 1), avg_cost))
                if (epoch + 1) % self.save_steps == 0:
                    saver.save(sess, self.checkpoints_path, global_step=(epoch + 1))

            self.show_variable()
            saver.save(sess, self.checkpoints_path, global_step=(self.training_epochs + 1))
            common_logger.info("Optimization Finished!")

    def test(self):
        """
        测试数据。
        :return:
        """
        pass

    def show_variable(self):
        """
        输出变量的结果。
        :return:
        """
        variable_names = [v.name for v in tf.trainable_variables()]
        values = self.sess.run(variable_names)
        for k, v in zip(variable_names, values):
            common_logger.info("Variable: {0} {1}, {2}".format(k, v, v.shape))


def plt_linear_model():
    """
    绘画线性模型的内容。
    :return:
    """
    train_X = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59,
                          2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
    train_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53,
                          1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])

    weight1, bias1 = 0.08040322, 1.7962127
    weight2, bias2 = 0.24164452, 0.8586517

    x = np.linspace(0, 10, 10)
    y1 = x * weight1 + bias1
    y2 = x * weight2 + bias2

    plt.figure()
    plt.scatter(train_X, train_Y)
    plt.plot(x, y1, 'r')
    plt.plot(x, y2, 'b')
    plt.show()


if __name__ == "__main__":
    # Training Data
    train_X = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59,
                          2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
    train_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53,
                          1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
    train_X = train_X.reshape((-1, 1))
    linearReg = LinearRegressionTF(train_X, train_Y, train_X.shape[1])
    linearReg.train()
    # plt_linear_model()
