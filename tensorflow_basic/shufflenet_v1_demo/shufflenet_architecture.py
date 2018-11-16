# -*- coding:utf-8 -*-

# ==============================================================================
# shufflenet网络模型的模型架构。
# ==============================================================================
import tensorflow as tf
import math
from tensorflow_basic.shufflenet_v1_demo.shufflenet_constant import *


def _channel_shuffle(x_data, groups):
    height, width, in_channels = x_data.shape.as_list()[1:]
    in_channels_per_group = int(in_channels/groups)

    # reshape
    shape = tf.stack([-1, height, width, groups, in_channels_per_group])
    x_data = tf.reshape(x_data, shape)

    # transpose
    x_data = tf.transpose(x_data, [0, 1, 2, 4, 3])

    # reshape
    shape = tf.stack([-1, height, width, in_channels])
    x_data = tf.reshape(x_data, shape)

    return x_data


def _mapping(
        X, is_training, num_classes=200,
        groups=3, dropout=0.5,
        complexity_scale_factor=0.75):
    """A ShuffleNet implementation.

    Arguments:
        X: A float tensor with shape [batch_size, image_height, image_width, 3].
        is_training: A boolean, whether the network is in the training mode.
        num_classes: An integer.
        groups: An integer, number of groups in group convolutions,
            only possible values are: 1, 2, 3, 4, 8.
        dropout: A floar number, dropout rate before the last linear layer.
        complexity_scale_factor: A floar number, to customize the network
            to a desired complexity you can apply a scale factor,
            in the original paper they are considering
            scale factor values: 0.25, 0.5, 1.0.
            It determines the width of the network.

    Returns:
        A float tensor with shape [batch_size, num_classes].
    """

    # 'out_channels' equals to second stage's number of output channels
    if groups == 1:
        out_channels = 144
    elif groups == 2:
        out_channels = 200
    elif groups == 3:
        out_channels = 240
    elif groups == 4:
        out_channels = 272
    elif groups == 8:
        out_channels = 384
    # all 'out_channels' are divisible by corresponding 'groups'

    # if you want you can decrease network's width
    out_channels = int(out_channels * complexity_scale_factor)

    with tf.variable_scope('features'):

        with tf.variable_scope('stage1'):

            with tf.variable_scope('conv1'):
                result = conv(X, 24, kernel=3, stride=FIRST_STRIDE)

            result = batch_norm(result, is_training)
            result = nonlinearity(result)
            # in the original paper they are not using batch_norm and relu here

            result = max_pooling(result)

        with tf.variable_scope('stage2'):

            with tf.variable_scope('unit1'):
                result = first_shufflenet_unit(
                    result, is_training, groups, out_channels
                )

            for i in range(N_SHUFFLE_UNITS[0]):
                with tf.variable_scope('unit' + str(i + 2)):
                    result = shufflenet_unit(result, is_training, groups)

            # number of channels in 'result' is out_channels

        with tf.variable_scope('stage3'):

            with tf.variable_scope('unit1'):
                result = shufflenet_unit(result, is_training, groups, stride=2)

            for i in range(N_SHUFFLE_UNITS[1]):
                with tf.variable_scope('unit' + str(i + 2)):
                    result = shufflenet_unit(result, is_training, groups)

            # number of channels in 'result' is 2*out_channels

        with tf.variable_scope('stage4'):

            with tf.variable_scope('unit1'):
                result = shufflenet_unit(result, is_training, groups, stride=2)

            for i in range(N_SHUFFLE_UNITS[2]):
                with tf.variable_scope('unit' + str(i + 2)):
                    result = shufflenet_unit(result, is_training, groups)

            # number of channels in 'result' is 4*out_channels

    with tf.variable_scope('classifier'):
        result = global_average_pooling(result)

        result = dropout(result, is_training, dropout)
        # in the original paper they are not using dropout here

        logits = affine(result, num_classes)

    return logits


def nonlinearity(x_data):
    """
    非线性化。
    :param x_data:
    :return:
    """
    return tf.nn.relu(x_data, name='ReLU')


def dropout(x_data, is_training, rate=0.5):
    """
    随机丢弃。
    :param x_data:
    :param is_training:
    :param rate:
    :return:
    """
    keep_prob = tf.constant(
        1.0 - rate, tf.float32,
        [], 'keep_prob'
    )
    result = tf.cond(
        is_training,
        lambda: tf.nn.dropout(x_data, keep_prob),
        lambda: tf.identity(x_data),
        name='dropout'
    )
    return result


def batch_norm(x_data, is_training):
    """
    Batch Normalization操作。
    :param x_data:
    :param is_training:
    :return:
    """
    return tf.layers.batch_normalization(
        x_data, scale=False, center=True,
        momentum=BATCH_NORM_MOMENTUM,
        training=is_training, fused=True
    )


def global_average_pooling(x_data):
    """
    全局平均池化操作。
    计算平均值，化简掉第1，2维，只保留第0维。
    :param x_data:
    :return:
    """
    return tf.reduce_mean(
        x_data, axis=[1, 2],
        name='global_average_pooling'
    )


def max_pooling(x_data):
    """
    最大池化操作。
    :param x_data:
    :return:
    """
    return tf.nn.max_pool(
        x_data, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME',
        name='max_pooling'
    )


def avg_pooling(x_data):
    """
    平均池化操作。
    :param x_data:
    :return:
    """
    return tf.nn.avg_pool(
        x_data, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME',
        name='avg_pooling'
    )


def conv(x_data, filters, kernel=3, stride=1):
    """
    卷积操作。
    :param x_data:
    :param filters:
    :param kernel:
    :param stride:
    :return:
    """
    in_channels = x_data.shape.as_list()[-1]

    # uniform initialization
    max_val = math.sqrt(6.0/in_channels)

    conv_kernel = tf.get_variable(
        'kernel', [kernel, kernel, in_channels, filters],
        tf.float32, tf.random_uniform_initializer(-max_val, max_val)
    )

    conv_bias = tf.get_variable(
        'bias', [filters], tf.float32,
        tf.zeros_initializer()
    )

    return tf.nn.bias_add(
        tf.nn.conv2d(x_data, conv_kernel, [1, stride, stride, 1], 'SAME'), conv_bias
    )


def group_conv(x_data, filters, groups, kernel=1, stride=1):
    """
    分组卷积操作。
    :param x_data:
    :param filters: 输出特征数量
    :param groups: 分组数量
    :param kernel: 卷积核尺寸
    :param stride: 卷积步长
    :return:
    """
    in_channels = x_data.shape.as_list()[3]
    in_channels_per_group = int(in_channels/groups)
    filters_per_group = int(filters/groups)

    # uniform initialization
    max_val = math.sqrt(6.0/in_channels_per_group)

    conv_kernel = tf.get_variable(
        'kernel', [kernel, kernel, in_channels_per_group, filters],
        tf.float32, tf.random_uniform_initializer(-max_val, max_val)
    )

    # split channels
    x_channel_splits = tf.split(x_data, [in_channels_per_group] * groups, axis=3)
    k_filter_splits = tf.split(conv_kernel, [filters_per_group]*groups, axis=3)

    results = []

    # do convolution for each split
    for i in range(groups):
        x_split = x_channel_splits[i]
        k_split = k_filter_splits[i]
        results += [tf.nn.conv2d(x_split, k_split, [1, stride, stride, 1], 'SAME')]

    return tf.concat(results, 3)


def depthwise_conv(x_data, kernel=3, stride=1):
    """
    depthwise 卷积操作。
    :param x_data:
    :param kernel:
    :param stride:
    :return:
    """
    in_channels = x_data.shape.as_list()[3]

    # uniform initialization
    max_val = math.sqrt(6.0/in_channels)

    depth_kernel = tf.get_variable(
        'depthwise_kernel', [kernel, kernel, in_channels, 1],
        tf.float32, tf.random_uniform_initializer(-max_val, max_val)
    )

    return tf.nn.depthwise_conv2d(x_data, depth_kernel, [1, stride, stride, 1], 'SAME')


def shufflenet_unit(x_data, is_training, groups=3, stride=1):

    in_channels = x_data.shape.as_list()[3]
    result = x_data

    with tf.variable_scope('g_conv_1'):
        result = group_conv(result, in_channels, groups)
        result = batch_norm(result, is_training)
        result = nonlinearity(result)

    with tf.variable_scope('channel_shuffle_2'):
        result = _channel_shuffle(result, groups)

    with tf.variable_scope('dw_conv_3'):
        result = depthwise_conv(result, stride=stride)
        result = batch_norm(result, is_training)

    with tf.variable_scope('g_conv_4'):
        result = group_conv(result, in_channels, groups)
        result = batch_norm(result, is_training)

    if stride < 2:
        result = tf.add(result, x_data)
    else:
        x_data = avg_pooling(x_data)
        result = tf.concat([result, x_data], 3)

    result = nonlinearity(result)
    return result


# first shufflenet unit is different from the rest
def first_shufflenet_unit(x_data, is_training, groups, out_channels):

    in_channels = x_data.shape.as_list()[3]
    result = x_data
    out_channels -= in_channels

    with tf.variable_scope('g_conv_1'):
        result = group_conv(result, out_channels, groups=1)
        result = batch_norm(result, is_training)
        result = nonlinearity(result)

    with tf.variable_scope('dw_conv_2'):
        result = depthwise_conv(result, stride=2)
        result = batch_norm(result, is_training)

    with tf.variable_scope('g_conv_3'):
        result = group_conv(result, out_channels, groups)
        result = batch_norm(result, is_training)

    x_data = avg_pooling(x_data)
    result = tf.concat([result, x_data], 3)
    result = nonlinearity(result)
    return result


def affine(x_data, size):
    input_dim = x_data.shape.as_list()[1]

    # uniform initialization
    max_val = math.sqrt(6.0/input_dim)

    W = tf.get_variable(
        'kernel', [input_dim, size], tf.float32,
        tf.random_uniform_initializer(-max_val, max_val)
    )

    b = tf.get_variable(
        'bias', [size], tf.float32,
        tf.zeros_initializer()
    )

    return tf.nn.bias_add(tf.matmul(x_data, W), b)


if __name__ == "__main__":
    pass
