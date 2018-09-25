# coding:utf-8

# ==============================================================================
# 提取ptb训练模型。
# ==============================================================================

import time

import numpy as np
import tensorflow as tf
from nlp_basic.ptb_demo.ptb_config import *
import nlp_basic.ptb_demo.ptb_reader as ptb_reader
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()


class PTBInput(object):
    """
    语言模型中输入处理。
    """
    def __init__(self, config, data, name=None):
        # 批次大小
        self.batch_size = batch_size = config.batch_size
        # LSTM 展开的步数
        self.num_steps = num_steps = config.num_steps
        # 每个epoch内训练需要的迭代次数
        self.epoch_size = (len(data) // batch_size - 1) // num_steps
        # 获取特征数据input_data、label数据targets
        self.input_data, self.targets = ptb_reader.ptb_producer(raw_data=data, batch_size=batch_size,
                                                                num_steps=num_steps, name=name)


class PTBModel(object):
    """
    语言模型。
    """
    # ============================================模型定义====================================================
    def __init__(self, is_training, config, input_):
        # 输入实例
        self._input = input_

        # LSTM展开的步数
        num_steps = input_.num_steps
        # 批次大小
        batch_size = input_.num_steps
        # LSTM的节点数
        size = config.hidden_size
        # 词汇表的大小
        vocab_size = config.vocab_size

        # -------------------------------------------LSTM单元---------------------------------------------------
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
        # 注意python中函数名字亦可作为变量
        attn_cell = lstm_cell
        # 如果在训练状态且keep_prob<1.0, 则添加dropout层
        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=config.keep_prob)
        # 多层LSTM的实现
        cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)
        # 将LSTM单元的初始化状态设置为0
        self._initial_state = cell.zero_state(batch_size, tf.float32)

        # "input"
        # 将相关的词向量转换分配给cpu
        with tf.device("/cpu:0"):
            # embedding矩阵，其行数与词数量相同，其列数（单词表达的维数）与LSTM的节点数相同
            embedding = tf.get_variable("embedding", [vocab_size, size], dtype=tf.float32)
            # 查询单词所对应的向量
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
            # 如果在训练阶段则添加dropout层
            if is_training and config.keep_prob < 1:
                inputs = tf.nn.dropout(inputs, config.keep_prob)

        # "output"
        outputs = []
        state = self._initial_state
        # 设置操作的名称, 便于变量共享
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                # 当从第2次循环开始，设置变量复用
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                # inputs有三个维度：[batch中对应的样本，样本中第几个单词，单词向量表达的维度],
                # inputs[:, time_step,:]代表所有样本中第time_step个单词
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        # -----------------------------------------Softmax------------------------------------------
        output = tf.reshape(tf.concat(outputs, 1), [-1, size])
        softmax_w = tf.get_variable(name="softmax_w", shape=[size, vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable(name="softmax_b", shape=[vocab_size], dtype=tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b

        # -----------------------------------------损失函数------------------------------------------
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [tf.reshape(input_.targets, [-1])],
                                                                  [tf.ones([batch_size*num_steps], dtype=tf.float32)])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state =state

        if not is_training:
            return

        # ---------------------------------------训练过程----------------------------------------------
        # 学习速率
        self._lr = tf.Variable(0.0, trainable=False)
        # 获取全部可训练的参数
        t_vars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, t_vars), config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, t_vars),
                                                   global_step=tf.contrib.framework.get_or_create_global_step())
        # 设置_new_lr来控制学习速率，同时定义_lr_update,使用tf.assign将_lr_update赋值给_lr
        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        """
        将模型进行训练
        :param session: 会话
        :param lr_value: 设置新的学习率
        :return:
        """
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    # ==========================================定义外部访问的API===========================================
    # "@property 装饰器可以将返回变量设置为只读，防止修改变量引发的问题"
    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


class TestConfig(object):
    """
    测试用模型， 参数配置都很小
    """
    # 网络中权重值的初始scale
    init_scale = 0.1
    # 学习速率的初始值
    learning_rate = 1.0
    # 梯度的最大范数
    max_grad_norm = 1
    # LSTM堆叠的层数
    num_layers = 1
    # LSTM反向传播的展开步数
    num_steps = 2
    # LSTM内的隐含节点数
    hidden_size = 2
    # 初始学习速率可训练的epoch数
    max_epoch = 1
    # 总共可训练的epoch数
    max_max_epoch = 1
    # dropout层保留节点的比例
    keep_prob = 1.0
    # 学习速率的衰减速度
    lr_decay = 0.5
    # 每个batch中样本的大小
    batch_size = 20
    # 词汇的大小
    vocab_size = 10000


def run_epoch(session, model, eval_op=None, verbose=False):
    """
    在给出的数据上跑模型
    :param session:
    :param model:
    :param eval_op:
    :param verbose:
    :return:
    """
    # 开始时间
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {"cost": model.cost, "final_state": model.final_state}
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        # enumerate 函数用于遍历序列中的元素以及它们的下标：
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            common_logger.info("{0:.3f} perplexity: {1:.3f} speed: {2:0>3} wps".format(
                    (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                     iters * model.batch_size / (time.time() - start_time))))

    return np.exp(costs / iters)


def main():
    """
    项目运行的API
    :return:
    """
    raw_data = ptb_reader.ptb_raw_data(ptb_data_path)
    train_data, valid_data, test_data, _ = raw_data
    config = SmallConfig()
    # 测试配置eval_config需要和训练配置一样, 这里将测试配置batch_size和num_steps修改为1
    eval_config = SmallConfig()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    # 设置图
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        # 训练过程
        with tf.name_scope("Train"):
            train_input = PTBInput(config=config, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = PTBModel(is_training=True, config=config, input_=train_input)

        # 验证过程
        with tf.name_scope("Valid"):
            valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                m_valid = PTBModel(is_training=False, config=config, input_=valid_input)

        # 测试过程
        with tf.name_scope("Test"):
            test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                m_test = PTBModel(is_training=False, config=eval_config, input_=test_input)

        # 创建训练的管理器sv, 并且用sv.managed_session创建默认session,再执行训练多个epoch数据循环
        sv = tf.train.Supervisor()
        with sv.managed_session() as session:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)
                common_logger.info("Epoch: {0} Learning rate: {1:.3f}".format(i + 1, session.run(m.lr)))

                train_perplexity = run_epoch(session, m, eval_op=m.train_op, verbose=True)
                common_logger.info("Epoch: {0} Train Perplexity: {1:.3f}".format(i + 1, train_perplexity))

                valid_perplexity = run_epoch(session, m_valid)
                common_logger.info("Epoch: {0} Valid Perplexity: {1:.3f}".format(i + 1, valid_perplexity))

            test_perplexity = run_epoch(session, m_test)
            common_logger.info("Test Perplexity: {1:.3f}".format(test_perplexity))


if __name__ == "__main__":
    main()
