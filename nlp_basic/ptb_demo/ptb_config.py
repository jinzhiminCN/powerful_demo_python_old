# _*_coding:utf-8_*_

# ==============================================================================
# 提取ptb训练的相关配置。
# ==============================================================================


ptb_data_path = "./data/"
ptb_save_path = ptb_data_path


class SmallConfig(object):
    """
    Small config.
    """
    # 网络中权重值的初始scale
    init_scale = 0.1
    # 学习率的初始值
    learning_rate = 1.0
    # 梯度的最大范数
    max_grad_norm = 5
    # LSTM堆叠的层数
    num_layers = 2
    # LSTM反向传播的展开步数
    num_steps = 20
    # LSTM内的隐含节点数
    hidden_size = 200
    # 初始学习率可训练的轮次
    max_epoch = 4
    # 总共可训练的轮次
    max_max_epoch = 13
    # dropout层保留节点的比例
    keep_prob = 1.0
    # 学习率的衰减速度
    lr_decay = 0.5
    # 每个batch中样本的数量
    batch_size = 20
    # 词汇总数
    vocab_size = 10000


class MediumConfig(object):
    """
    Medium config.
    """
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000


class LargeConfig(object):
    """
    Large config.
    """
    init_scale = 0.004
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000


class XLargeConfig(object):
    """
    Large config.
    """
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 3
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000


class XXLargeConfig(object):
    """
    Large config.
    """
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 4
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000


def get_config():
    """
    获取config。
    :return:
    """
    if FLAGS.model == "small":
        return SmallConfig()
    elif FLAGS.model == "medium":
        return MediumConfig()
    elif FLAGS.model == "large":
        return LargeConfig()
    elif FLAGS.model == "xlarge":
        return XLargeConfig()
    elif FLAGS.model == "xxlarge":
        return XXLargeConfig()
    else:
        raise ValueError("Invalid model: {0}".format(FLAGS.model))