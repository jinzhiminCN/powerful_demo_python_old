# -*- coding:utf-8 -*-

# ==============================================================================
# tensorflow简单网络结构。
# ==============================================================================
from tensorflow_basic.base_dnn_tf import *


class BasicDNN(BaseDNN):
    """
    最简单的一层神经网络结构。
    """
    def __init__(self, input_dim, output_dim):
        """
        初始化神经网络的数据。
        :param input_dim: 输入数据的维度
        :param output_dim: 输出结果的维度
        """
        # 网络结构参数
        self.x_input = None
        self.y_label = None
        self.weight = None
        self.bias = None
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 训练需要的超参数
        self.learning_rate = 1e-5
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
        self.tf_logs_path = os.path.join(com_config.TF_MODEL_LOGS_DIR, "basicDNN")
        self.checkpoints_path = os.path.join(com_config.TF_MODEL_CHECKPOINT_DIR, "basicDNN")

        # 变量和参数初始化
        self.init_network()

    @staticmethod
    def name():
        """
        网络结构名称。
        :return:
        """
        return "Basic Deep Neural Network(DNN)"

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
        self.x_input = tf.placeholder(tf.float32, [None, self.input_dim], name="x_input")
        # 模型的输入y值
        self.y_label = tf.placeholder(tf.float32, [None, self.output_dim], name="y_label")

    def variables(self):
        """
        使用到的变量。
        :return:
        """
        with tf.name_scope("Layer"):
            # 权重变量
            self.weight = tf.Variable(tf.zeros([self.input_dim, self.output_dim]), name='Weights')
            # 偏置变量
            self.bias = tf.Variable(tf.zeros([self.output_dim]), name='Bias')

    def inference(self):
        """
        网络结构生成。
        :return:
        """
        # 计算预测值
        self.y_value = tf.matmul(self.x_input, self.weight) + self.bias
        self.y_predict = tf.nn.softmax(self.y_value)

    def loss_function(self):
        """
        损失函数设置。
        :return:
        """
        # 1. 直接计算交叉熵
        self.loss = tf.reduce_mean(tf.reduce_sum(-self.y_label * tf.log(self.y_predict)))
        # 2. 差平方求和再平均
        # self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.y_label - self.y_predict)))
        # 3. 计算softmax交叉熵
        # softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_label, logits=self.y_value)
        # self.loss = tf.reduce_mean(tf.reduce_sum(softmax_cross_entropy))

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
        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        # 2. Adam
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

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
        super().train_mnist()
        self.show_variable()

    def show_variable(self):
        """
        输出变量的结果。
        :return:
        """
        variable_names = [v.name for v in tf.trainable_variables()]
        values = self.sess.run(variable_names)
        for k, v in zip(variable_names, values):
            common_logger.info("Variable: {0} {1}, {2}".format(k, v, v.shape))

    @staticmethod
    def show_variable_detail():
        """
        显示变量细节。
        :return:
        """
        checkpoint_path = os.path.join(com_config.TF_MODEL_CHECKPOINT_DIR, "basicDNN-51")
        reader = tf.train.NewCheckpointReader(checkpoint_path)
        all_variables = reader.get_variable_to_shape_map()
        common_logger.info(all_variables)
        w1 = reader.get_tensor("Layer/Weights")
        b1 = reader.get_tensor("Layer/Bias")
        for i in range(784):
            common_logger.info("Weight_{0:0>3}:{1}".format(i, w1[i]))
        common_logger.info("Bias:{0}".format(b1))


if __name__ == "__main__":
    basicDNN = BasicDNN(n_input, n_classes)
    basicDNN.train_mnist()
    # BasicDNN.show_variable_detail()

