# -*- coding:utf-8 -*-

# ==============================================================================
# tensorflow 深度神经网络基类。
# @abc.abstractmethod修饰的方法子类必须重新实现。
# ==============================================================================
import abc
import tensorflow as tf
import os
import config.common_config as com_config
from tensorflow.examples.tutorials.mnist import input_data
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()
# mnist数据
mnist = input_data.read_data_sets(com_config.MNIST_DIR, one_hot=True)
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)


class BaseDNN(abc.ABC):
    """
    深度神经网络基类。
    """
    def __init__(self, input_dim, output_dim):
        """
        初始化网络数据
        :param input_dim:
        :param output_dim:
        """
        # 网络结构参数
        self.x_input = None
        self.y_label = None
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 训练需要的超参数
        self.learning_rate = 1e-2
        self.training_epochs = 50
        self.batch_size = 100
        self.display_steps = 1
        self.save_steps = 1

        # 预测优化函数
        self.y_predict = None
        self.y_value = None
        self.loss = None
        self.accuracy = None
        self.optimizer = None

        # sess会话
        self.sess = None

        # 初始化器
        self.initializer = None
        self.merged_summary_op = None

        # 目录初始化
        self.tf_logs_path = os.path.join(com_config.TF_MODEL_LOGS_DIR, "BaseDNN")
        self.checkpoints_path = os.path.join(com_config.TF_MODEL_CHECKPOINT_DIR, "BaseDNN")

        # 变量和参数初始化
        self.init_network()

    @staticmethod
    def name():
        """
        网络结构名称。
        :return:
        """
        return "Base Deep Neural Network(BaseDNN)"

    def init_network(self):
        """
        初始化网络结构信息。
        :return:
        """
        self.placeholders()
        self.inference()
        self.loss_function()
        self.evaluate_function()
        self.summary()
        self.solver()
        # 初始化变量的操作应该放在最后
        self.init_variables()

    @abc.abstractmethod
    def placeholders(self):
        """
        使用到的占位符。
        :return:
        """
        pass

    def init_variables(self):
        """
        初始化变量。
        :return:
        """
        self.initializer = tf.global_variables_initializer()

    @abc.abstractmethod
    def inference(self):
        """
        网络结构生成。
        :return:
        """
        pass

    @abc.abstractmethod
    def loss_function(self):
        """
        损失函数设置。
        :return:
        """
        pass

    def evaluate_function(self):
        """
        评价函数设置。
        :return:
        """
        pass

    @abc.abstractmethod
    def solver(self):
        """
        求解器。
        :return:
        """
        pass

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
        pass

    def train_mnist(self):
        """
        训练mnist数据
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
                total_batch = int(mnist.train.num_examples / self.batch_size)
                # 循环运行所有批次数据
                for i_batch in range(total_batch):
                    batch_xs, batch_ys = mnist.train.next_batch(self.batch_size)

                    # 执行优化、损失函数、准确率
                    _, cost_value, accuracy_value, summary = \
                        sess.run([self.optimizer, self.loss, self.accuracy, self.merged_summary_op],
                                 feed_dict={self.x_input: batch_xs, self.y_label: batch_ys})

                    # 在summary_writer中记录相应的训练过程
                    summary_writer.add_summary(summary, epoch * total_batch + i_batch)
                    # 计算平均损失
                    avg_cost += cost_value / total_batch

                    common_logger.info("Epoch: {0:0>4}_{1:0>4} cost={2:.9f} accuracy={3:.9f}"
                                       .format((epoch + 1), i_batch, cost_value, accuracy_value))

                # 记录每轮迭代的中间结果
                if (epoch + 1) % self.display_steps == 0:
                    common_logger.info("Epoch: {0:0>4} cost={1:.9f}".format((epoch + 1), avg_cost))
                if (epoch + 1) % self.save_steps == 0:
                    saver.save(sess, self.checkpoints_path, global_step=(epoch + 1))

            # 记录训练的最终结果
            saver.save(sess, self.checkpoints_path, global_step=(self.training_epochs + 1))
            common_logger.info("Optimization Finished!")

            # 测试模型，计算测试数据的准确率
            test_accuracy = self.accuracy.eval(
                {self.x_input: mnist.test.images, self.y_label: mnist.test.labels})

            common_logger.info("Test Accuracy:{0}".format(test_accuracy))
