# -*- coding:utf-8 -*-

# ==============================================================================
# tensorflow版本的逻辑回归模型。
# ==============================================================================
import tensorflow as tf
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import config.common_config as com_config
from util.log_util import LoggerUtil
import machine_learning.logistic_regression_demo as ml_logistic_reg
from util.tensorflow_util import TensorFlowUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()


class LogisticRegressionTF(object):
    """
    逻辑回归模型TensorFlow版本。
    """
    def __init__(self, x_data, y_data, input_dim):
        """
        初始化逻辑回归模型。
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
        self.theta = None

        # 训练需要的超参数
        self.learning_rate = 1e-2
        self.training_epochs = 1000
        self.batch_size = 100
        self.display_steps = 50
        self.save_steps = 100

        # 预测优化函数
        self.y_value = None
        self.y_predict = None
        self.loss = None
        self.accuracy = None
        self.optimizer = None

        # sess会话
        self.sess = None

        # 初始化器
        self.initializer = None
        self.merged_summary_op = None

        # 目录初始化
        self.tf_logs_path = os.path.join(com_config.TF_MODEL_LOGS_DIR, "LogisticRegressionTF")
        self.checkpoints_path = os.path.join(com_config.TF_MODEL_CHECKPOINT_DIR, "LogisticRegressionTF")

        # 变量和参数初始化
        self.init_network()

    @staticmethod
    def name(self):
        """
        结构名称。
        :return:
        """
        return "Logistic Regression TensorFlow(LR)"

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
        self.weight = tf.Variable(tf.ones([self.input_dim, 1]), name='Weights')
        # 2. 权重初始化为随机值
        # self.weight = tf.Variable(tf.random_uniform([self.input_dim, 1], -1.0, 1.0), name='Weights')

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
        self.y_predict = tf.sigmoid(self.y_value)

    def loss_function(self):
        """
        损失函数设置。
        :return:
        """
        self.loss = tf.reduce_mean(
            -self.y_input*tf.log(self.y_predict) - (1-self.y_input)*tf.log(1-self.y_predict))

    def evaluate_function(self):
        """
        评价函数设置。
        :return:
        """
        y_class = tf.cast(self.y_predict > 0.5, tf.float32)
        correct_prediction = tf.equal(y_class, self.y_input)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def solver(self):
        """
        求解器。
        :return:
        """
        # 1. Gradient Descent
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        gradient = self.optimizer.compute_gradients(self.loss)
        common_logger.info(gradient)
        self.optimizer = self.optimizer.apply_gradients(gradient)

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
                    _, cost_value, accuracy_value, summary = \
                        sess.run([self.optimizer, self.loss, self.accuracy, self.merged_summary_op],
                                 feed_dict={self.x_input: batch_xs, self.y_input: batch_ys})

                    # 在summary_writer中记录相应的训练过程
                    summary_writer.add_summary(summary, epoch * total_batch + i)
                    # 计算平均损失
                    avg_cost += cost_value / total_batch

                    common_logger.info("Epoch: {0:0>4}_{1:0>4} cost={2:.9f}, accuracy={3:.9f}".format(
                        (epoch + 1), i, cost_value, accuracy_value))

                # 记录每轮迭代的中间结果
                if (epoch + 1) % self.display_steps == 0:
                    common_logger.info("Epoch: {0:0>4} cost={1:.9f}".format((epoch + 1), avg_cost))
                if (epoch + 1) % self.save_steps == 0:
                    saver.save(sess, self.checkpoints_path, global_step=(epoch + 1))

            self.show_variable()
            saver.save(sess, self.checkpoints_path, global_step=(self.training_epochs + 1))
            common_logger.info("Optimization Finished!")

    def show_variable(self):
        """
        输出变量的结果。
        :return:
        """
        variable_names = [v.name for v in tf.trainable_variables()]
        values = self.sess.run(variable_names)
        for k, v in zip(variable_names, values):
            common_logger.info("Variable: {0} {1}, {2}".format(k, v, v.shape))
            if k == "Weights:0":
                current_weight = v
            elif k == "Bias:0":
                current_bias = v
        self.theta = current_weight
        self.theta[0] += current_bias
        common_logger.info("theta:{0}".format(self.theta))


def logistic_regression_model():
    """
    逻辑回归模型。
    :return:
    """
    x_data, y_data = ml_logistic_reg.load_data_set()
    x_data = np.array(x_data)[:, 1:]
    y_data = np.array(y_data).transpose()
    y_data = y_data.reshape((-1, 1))

    w = tf.Variable(tf.zeros([2, 1]))
    b = tf.Variable(tf.zeros([1, 1]))
    w = tf.cast(w, tf.float64)
    b = tf.cast(b, tf.float64)
    # y = 1 / (1 + tf.exp(-tf.matmul(x_data, w) + b))
    y_pred = tf.nn.softmax(tf.matmul(x_data, w) + b)
    loss = tf.reduce_mean(-tf.reduce_sum(y_data*tf.log(y_pred), reduction_indices=1))

    optimizer = tf.train.GradientDescentOptimizer(0.001)
    train = optimizer.minimize(loss)

    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)
    common_logger.info("y_pred:{0}".format(sess.run(y_pred)))
    for step in range(500):
        sess.run(train)
        if step % 100 == 0:
            common_logger.info("step:{0}, weight:{1}, bias:{2}"
                               .format(step, sess.run(w).flatten(), sess.run(b).flatten()))

    common_logger.info("Coefficients of tensorflow (input should be standardized): K={0}, b={1}".format(
        sess.run(w).flatten(), sess.run(b).flatten()))

    weight = sess.run(w).flatten().tolist()
    bias = sess.run(b).flatten().tolist()
    theta = bias + weight

    theta_list = [theta]
    ml_logistic_reg.plot_best_fit(theta_list)


def test_train():
    """
    测试训练数据的过程。
    :return:
    """
    x_data, y_data = ml_logistic_reg.load_data_set()
    x_data = np.array(x_data)
    y_data = np.array(y_data).transpose()
    input_dim = x_data.shape[1]
    scaler = preprocessing.StandardScaler().fit(x_data)
    x_data_standard = scaler.transform(x_data)

    logistic_regression_tf = LogisticRegressionTF(x_data_standard, y_data, input_dim)
    logistic_regression_tf.train()
    theta_list = [logistic_regression_tf.theta]
    ml_logistic_reg.plot_best_fit(theta_list)


def logistic_reg_model():
    """
    逻辑回归模型测试。
    :return:
    """
    # x_train = [[1.0, 2.0], [2.0, 1.0], [2.0, 3.0], [3.0, 5.0], [1.0, 3.0],
    #            [4.0, 2.0], [7.0, 3.0], [4.0, 5.0], [11.0, 3.0], [8.0, 7.0]]
    # y_train = [1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    # x_train = [[1.0] + val for val in x_train]
    # x_data, y_data = x_train, y_train
    x_data, y_data = ml_logistic_reg.load_data_set()

    x_data = np.array(x_data)[:, 1:]
    y_data = np.array(y_data).transpose()

    # 学习率
    alpha = 0.01
    # 最大迭代次数
    max_iters = 10000

    common_logger.info("x_data:{0}".format(x_data))
    common_logger.info("y_data:{0}".format(y_data))
    common_logger.info("y_data shape:{0}".format(y_data.size))

    w = tf.Variable(tf.zeros([2, 1]))
    b = tf.Variable(tf.zeros([1, 1]))
    w = tf.cast(w, tf.float64)
    b = tf.cast(b, tf.float64)
    y_predict = 1 / (1 + tf.exp(-tf.matmul(x_data, w) + b))
    # y_predict = tf.sigmoid(-tf.matmul(x_data, w) + b)
    loss = tf.reduce_mean(- y_data.reshape(-1, 1) * tf.log(y_predict)
                          - (1 - y_data.reshape(-1, 1)) * tf.log(1 - y_predict))

    init = tf.initialize_all_variables()
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(init)

    for step in range(max_iters):
        sess.run(train)
        common_logger.info("step:{0}, weight:{1}, bias:{2}"
                           .format(step, sess.run(w).flatten(), sess.run(b).flatten()))

    weight = sess.run(w).flatten().tolist()
    bias = sess.run(b).flatten().tolist()
    theta = bias + weight

    theta_list = [theta]
    plot_best_fit(theta_list, x_data, y_data)


def plot_best_fit(theta_list, data_mat, label_mat):
    """
    根据训练好的theta绘图。
    :param theta_list:
    :param data_mat:
    :param label_mat:
    :return:
    """
    # 加载数据
    data_arr = np.array(data_mat)
    n_datas = np.shape(data_arr)[0]
    max_val = np.max(data_arr)
    min_val = np.min(data_arr)

    # x, y坐标
    x_cord1 = []
    y_cord1 = []
    x_cord2 = []
    y_cord2 = []

    # 将数据按真实标签进行分类
    for i in range(n_datas):
        if int(label_mat[i]) == 1:
            x_cord1.append(data_arr[i, 0])
            y_cord1.append(data_arr[i, 1])
        else:
            x_cord2.append(data_arr[i, 0])
            y_cord2.append(data_arr[i, 1])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_cord1, y_cord1, s=30, c='red', marker='s')
    ax.scatter(x_cord2, y_cord2, s=30, c='blue')
    # 生成x的取值-3.0--3.0, 增量为0.1
    x1 = np.arange(min_val - 1, max_val + 1, 0.1)
    # 根据y=Θ0+Θ1x1+Θ2x2, 求y=0时的x2
    for theta in theta_list:
        x2 = (-theta[0] - theta[1] * x1) / theta[2]
        ax.plot(x1, x2.T)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


if __name__ == "__main__":
    # test_train()
    # logistic_regression_model()
    logistic_reg_model()
    pass
