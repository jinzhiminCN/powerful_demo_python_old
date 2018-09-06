# -*- coding:utf-8 -*-

# ==============================================================================
# tensorflow基本网络结构(包含隐藏层)。
# ==============================================================================
from tensorflow_basic.base_dnn_tf import *


class BasicHiddenDNN(BaseDNN):
    """
    最简单的包含隐藏层的神经网络结构。
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
    def name():
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

    def placeholders(self):
        """
        使用到的占位符。
        :return:
        """
        # 模型的输入x值
        self.x_input = tf.placeholder(tf.float32, [None, self.input_dim], name="x_input")
        # 模型的输入y值
        self.y_label = tf.placeholder(tf.float32, [None, self.output_dim], name="y_label")

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


def mnist_single_perceptron_model():
    """
    mnist单隐藏层感知器模型。
    :return:
    """
    # 参数和占位符
    x_input = tf.placeholder(tf.float32, [None, 784])
    y_label = tf.placeholder(tf.float32, [None, 10])
    w1 = tf.Variable(tf.random_normal([784, 256]))
    b1 = tf.Variable(tf.random_normal([256]))
    w2 = tf.Variable(tf.random_normal([256, 10]))
    b2 = tf.Variable(tf.random_normal([10]))

    # 构建网络
    lay1 = tf.nn.relu(tf.matmul(x_input, w1) + b1)
    y = tf.add(tf.matmul(lay1, w2), b2)

    # 损失函数和准确率
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_label))
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 优化函数
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # 执行训练
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    for index in range(10000):
        # 分批训练
        batch_xs, batch_ys = mnist.train.next_batch(100)
        _, cost_value, accuracy_value = sess.run([train_step, cross_entropy, accuracy],
                                                 feed_dict={x_input: batch_xs, y_label: batch_ys})
        common_logger.info("Epoch: {0:0>4} cost={1:.9f} accuracy={2:.9f}"
                           .format(index, cost_value, accuracy_value))

    # 训练数据的准确率
    train_accuracy = sess.run(accuracy, feed_dict={x_input: mnist.train.images,
                                                   y_label: mnist.train.labels})
    common_logger.info("Train Accuracy:{0:.9f}".format(train_accuracy))

    # 测试数据的准确率
    test_accuracy = sess.run(accuracy, feed_dict={x_input: mnist.test.images,
                                                  y_label: mnist.test.labels})
    common_logger.info("Test Accuracy:{0:.9f}".format(test_accuracy))


def mnist_multi_perceptron_model():
    """
    mnist多隐藏层感知器模型。
    :return:
    """
    # 参数和占位符
    x_input = tf.placeholder(tf.float32, [None, 784])
    y_label = tf.placeholder(tf.float32, [None, 10])
    w1 = tf.Variable(tf.random_normal([784, 256]))
    b1 = tf.Variable(tf.random_normal([256]))
    w2 = tf.Variable(tf.random_normal([256, 256]))
    b2 = tf.Variable(tf.random_normal([256]))
    w3 = tf.Variable(tf.random_normal([256, 10]))
    b3 = tf.Variable(tf.random_normal([10]))

    # 构建网络
    lay1 = tf.nn.relu(tf.add(tf.matmul(x_input, w1), b1))
    lay2 = tf.nn.relu(tf.add(tf.matmul(lay1, w2), b2))
    y = tf.add(tf.matmul(lay2, w3), b3)

    # 损失函数和准确率
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_label))
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 优化函数
    train_step = tf.train.GradientDescentOptimizer(0.0095).minimize(cross_entropy)

    # 执行训练
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    for index in range(10000):
        # 分批训练
        batch_xs, batch_ys = mnist.train.next_batch(100)
        _, cost_value, accuracy_value = sess.run([train_step, cross_entropy, accuracy],
                                                 feed_dict={x_input: batch_xs, y_label: batch_ys})
        common_logger.info("Epoch: {0:0>4} cost={1:.9f} accuracy={2:.9f}"
                           .format(index, cost_value, accuracy_value))

    # 训练数据的准确率
    train_accuracy = sess.run(accuracy, feed_dict={x_input: mnist.train.images,
                                                   y_label: mnist.train.labels})
    common_logger.info("Train Accuracy:{0:.9f}".format(train_accuracy))

    # 测试数据的准确率
    test_accuracy = sess.run(accuracy, feed_dict={x_input: mnist.test.images,
                                                  y_label: mnist.test.labels})
    common_logger.info("Test Accuracy:{0:.9f}".format(test_accuracy))


if __name__ == "__main__":
    basicDNN = BasicHiddenDNN(n_input, n_classes)
    basicDNN.train_mnist()
    # mnist_multi_perceptron_model()
    pass

