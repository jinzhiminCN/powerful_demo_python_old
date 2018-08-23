# -*- coding:utf-8 -*-

# ==============================================================================
# tensorflow基本网络结构(包含一层隐藏层)。
# ==============================================================================
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


class BasicHiddenDNN(object):
    """
    最简单的一层神经网络结构。
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
        self.weight = None
        self.bias = None
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
        self.tf_logs_path = os.path.join(com_config.TF_MODEL_LOGS_DIR, "BasicHiddenDNN")
        self.checkpoints_path = os.path.join(com_config.TF_MODEL_CHECKPOINT_DIR, "BasicHiddenDNN")

        # 变量和参数初始化
        self.init_network()

    @staticmethod
    def name(self):
        """
        网络结构名称。
        :return:
        """
        return "Basic Hidden Network(BasicHiddenDNN)"

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
        self.x_input = tf.placeholder(tf.float32, [None, self.input_dim], name="x_input")
        # 模型的输入y值
        self.y_label = tf.placeholder(tf.float32, [None, self.output_dim], name="y_label")

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
        hidden_num1 = 256
        hidden_layer1 = BasicHiddenDNN.full_connect(self.x_input, self.input_dim, hidden_num1, "weight1", "bias1")
        hidden_layer1 = tf.nn.relu(hidden_layer1)
        self.y_value = BasicHiddenDNN.full_connect(hidden_layer1, hidden_num1, self.output_dim, "weight2", "bias2")
        self.y_predict = tf.nn.softmax(self.y_value)

    @staticmethod
    def full_connect(input_tensor, input_dim, output_dim, weight_name, bias_name):
        """
        全连接网络。
        :param input_tensor:
        :param input_dim:
        :param output_dim:
        :param weight_name:
        :param bias_name:
        :return:
        """
        weight = tf.Variable(tf.random_normal([input_dim, output_dim]), name=weight_name)
        bias = tf.Variable(tf.random_normal([output_dim]), name=bias_name)
        result = tf.matmul(input_tensor, weight) + bias
        return result

    def loss_function(self):
        """
        损失函数设置。
        :return:
        """
        # 1. 直接计算交叉熵
        # self.loss = tf.reduce_mean(tf.reduce_sum(-self.y_label * tf.log(self.y_predict)))
        # 2. 差平方求和再平均
        # self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.y_label - self.y_predict)))
        # 3. 计算softmax交叉熵
        softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_label, logits=self.y_value)
        self.loss = tf.reduce_mean(tf.reduce_sum(softmax_cross_entropy))

    def evaluate_function(self):
        """
        评价函数设置。
        :return:
        """
        self.accuracy = tf.equal(tf.argmax(self.y_predict, 1), tf.argmax(self.y_label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.accuracy, tf.float32))

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
            # 初始化
            sess.run(self.initializer)
            # 训练数据

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
                for i in range(total_batch):
                    batch_xs, batch_ys = mnist.train.next_batch(self.batch_size)

                    # 执行优化、损失函数、准确率
                    _, cost_value, accuracy_value, summary = \
                        sess.run([self.optimizer, self.loss, self.accuracy, self.merged_summary_op],
                                 feed_dict={self.x_input: batch_xs, self.y_label: batch_ys})

                    # 在summary_writer中记录相应的训练过程
                    summary_writer.add_summary(summary, epoch * total_batch + i)
                    # 计算平均损失
                    avg_cost += cost_value / total_batch

                    common_logger.info("Epoch: {0:0>4}_{1:0>4} cost={2:.9f} accuracy={3:.9f}".format(
                        (epoch + 1), i, cost_value, accuracy_value))

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

            common_logger.info("Accuracy:{0}".format(test_accuracy))

    def test(self):
        """
        测试数据。
        :return:
        """
        pass


def simple_mnist_model():
    """
    简单模型示例
    :return:
    """
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    w1 = tf.Variable(tf.random_normal([784, 256]))
    b1 = tf.Variable(tf.random_normal([256]))
    w2 = tf.Variable(tf.random_normal([256, 10]))
    b2 = tf.Variable(tf.random_normal([10]))
    lay1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    y = tf.add(tf.matmul(lay1, w2), b2)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    for index in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        _, cost_value, accuracy_value = sess.run([train_step, cross_entropy, accuracy],
                                                 feed_dict={x: batch_xs, y_: batch_ys})
        common_logger.info("Epoch: {0:0>4}_{1:0>4} cost={2:.9f} accuracy={3:.9f}".format(
            (index + 1), index, cost_value, accuracy_value))

    common_logger.info(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                     y_: mnist.test.labels}))


if __name__ == "__main__":
    basicDNN = BasicHiddenDNN(n_input, n_classes)
    basicDNN.train_mnist()
    # simple_mnist_model()


