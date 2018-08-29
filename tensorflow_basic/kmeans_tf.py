# -*- coding:utf-8 -*-

# ==============================================================================
# tensorflow版本的kmeans模型。
# ==============================================================================
import tensorflow as tf
import os
import math
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.factorization import KMeans
import config.common_config as com_config
from util.log_util import LoggerUtil


# 日志器
common_logger = LoggerUtil.get_common_logger()
# mnist数据
mnist = input_data.read_data_sets(com_config.MNIST_DIR, one_hot=True)


class KMeansTF(object):
    """
    KMeans模型TensorFlow版本。
    """
    def __init__(self, x_data, y_data, input_dim, output_dim, cluster_num):
        """
        初始化线性回归模型。
        :param x_data: 输入x数据
        :param y_data: 输入y数据
        :param input_dim: 输入x的维度
        :param output_dim: 输入y的维度
        :param cluster_num: 聚类簇的数量
        """
        # 网络结构参数
        self.x_input = None
        self.y_input = None
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cluster_num = cluster_num
        self.x_data = x_data
        self.y_data = y_data

        # 训练需要的超参数
        self.training_epochs = 100
        self.display_steps = 50

        # 预测优化函数
        self.cluster_idx = None
        self.avg_distance = None
        self.train_op = None
        self.init_op = None

        # sess会话
        self.sess = None

        # 初始化器
        self.initializer = None
        self.merged_summary_op = None

        # 目录初始化
        self.tf_logs_path = os.path.join(com_config.TF_MODEL_LOGS_DIR, "KMeansTF")
        self.checkpoints_path = os.path.join(com_config.TF_MODEL_CHECKPOINT_DIR, "KMeansTF")

        # 变量和参数初始化
        self.init_network()

    @staticmethod
    def name(self):
        """
        结构名称。
        :return:
        """
        return "KMeans TensorFlow(LR)"

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
        # 模型的输入x值
        self.x_input = tf.placeholder(tf.float32, shape=[None, self.input_dim], name="x_input")
        # 模型的输入y值
        self.y_input = tf.placeholder(tf.float32, shape=[None, self.output_dim], name="y_input")

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
        # K-Means Parameters
        kmeans = KMeans(inputs=self.x_input, num_clusters=self.cluster_num, distance_metric='cosine',
                        use_mini_batch=True)

        # Build KMeans graph
        training_graph = kmeans.training_graph()

        if len(training_graph) > 6:  # Tensorflow 1.4+
            (all_scores, cluster_idx, scores, cluster_centers_initialized,
             cluster_centers_var, init_op, train_op) = training_graph
        else:
            (all_scores, cluster_idx, scores, cluster_centers_initialized,
             init_op, train_op) = training_graph

        self.cluster_idx = cluster_idx[0]  # fix for cluster_idx being a tuple
        self.avg_distance = tf.reduce_mean(scores)
        self.train_op = train_op
        self.init_op = init_op

    def train_mnist(self):
        """
        训练网络参数。
        :return:
        """
        with tf.Session() as sess:
            self.sess = sess
            sess.run(self.initializer, feed_dict={self.x_input: self.x_data})
            sess.run(self.init_op, feed_dict={self.x_input: self.x_data})

            # Training
            for i in range(1, self.training_epochs + 1):
                _, dist, idx = sess.run([self.train_op, self.avg_distance, self.cluster_idx],
                                        feed_dict={self.x_input: self.x_data})
                if i % 10 == 0 or i == 1:
                    common_logger.info("Step {0}, Avg Distance: {1:>6}".format(i, dist))

            counts = np.zeros(shape=(self.cluster_num, self.output_dim))
            for i in range(len(idx)):
                counts[idx[i]] += mnist.train.labels[i]
            # Assign the most frequent label to the centroid
            labels_map = [np.argmax(c) for c in counts]
            common_logger.info(labels_map)
            labels_map = tf.convert_to_tensor(labels_map)

            # Lookup: centroid_id -> label
            cluster_label = tf.nn.embedding_lookup(labels_map, self.cluster_idx)
            correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(self.y_input, 1), tf.int32))
            accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # Test Model
            test_x, test_y = mnist.test.images, mnist.test.labels
            test_accuracy = sess.run(accuracy_op, feed_dict={self.x_input: test_x, self.y_input: test_y})
            common_logger.info("Test Accuracy:{0}".format(test_accuracy))


def kmeans_model():
    """
    kmeans模型。
    :return:
    """
    train_x, train_y = mnist.train.images, mnist.train.labels

    num_steps = 100  # Total steps to train
    batch_size = 1024  # The number of samples per batch
    k = 30  # The number of clusters
    num_classes = 10  # The 10 digits
    num_features = 784  # Each image is 28x28 pixels

    # Input images
    x_input = tf.placeholder(tf.float32, shape=[None, num_features])
    y_input = tf.placeholder(tf.float32, shape=[None, num_classes])

    # K-Means Parameters
    kmeans = KMeans(inputs=x_input, num_clusters=k, distance_metric='cosine',
                    use_mini_batch=True)

    # Build KMeans graph
    training_graph = kmeans.training_graph()

    if len(training_graph) > 6:  # Tensorflow 1.4+
        (all_scores, cluster_idx, scores, cluster_centers_initialized,
         cluster_centers_var, init_op, train_op) = training_graph
    else:
        (all_scores, cluster_idx, scores, cluster_centers_initialized,
         init_op, train_op) = training_graph

    cluster_idx = cluster_idx[0]  # fix for cluster_idx being a tuple
    avg_distance = tf.reduce_mean(scores)

    # Initialize the variables (i.e. assign their default value)
    init_vars = tf.global_variables_initializer()

    # Start TensorFlow session
    sess = tf.Session()

    # Run the initializer
    sess.run(init_vars, feed_dict={x_input: train_x})
    sess.run(init_op, feed_dict={x_input: train_x})

    # Training
    for i in range(1, num_steps + 1):
        _, dist, idx = sess.run([train_op, avg_distance, cluster_idx],
                                feed_dict={x_input: train_x})
        if i % 10 == 0 or i == 1:
            common_logger.info("Step {0}, Avg Distance: {1:>6}".format(i, dist))

    counts = np.zeros(shape=(k, num_classes))
    for i in range(len(idx)):
        counts[idx[i]] += mnist.train.labels[i]
    # Assign the most frequent label to the centroid
    labels_map = [np.argmax(c) for c in counts]
    common_logger.info(labels_map)
    labels_map = tf.convert_to_tensor(labels_map)

    # Lookup: centroid_id -> label
    cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)
    correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(y_input, 1), tf.int32))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Test Model
    test_x, test_y = mnist.test.images, mnist.test.labels
    test_accuracy = sess.run(accuracy_op, feed_dict={x_input: test_x, y_input: test_y})
    common_logger.info("Test Accuracy:{0}".format(test_accuracy))


def test_train():
    """
    测试训练数据的过程。
    :return:
    """
    train_x, train_y = mnist.train.images, mnist.train.labels

    k = 30
    num_classes = 10
    num_features = 784

    kmeans = KMeansTF(train_x, train_y, num_features, num_classes, k)
    kmeans.train_mnist()


if __name__ == "__main__":
    test_train()
    # kmeans_model()
    pass
