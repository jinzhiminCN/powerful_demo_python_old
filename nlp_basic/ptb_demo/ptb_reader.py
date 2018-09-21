# _*_coding:utf-8_*_

# ==============================================================================
# ptb读取数据的相关方法。
# ==============================================================================


import collections
import os

import numpy as np
import tensorflow as tf
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()
ptb_data_path = './data'


def _read_words(filename):
    """
    读取单词。
    :param filename:
    :return:
    """
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
    """
    构建词典，单词和单词id的对应关系。
    :param filename:
    :return:
    """
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def _file_to_word_ids(filename, word_to_id):
    """
    文件内容转换为单词id的序列。
    :param filename:
    :param word_to_id:
    :return:
    """
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
    """
    Load PTB raw data from data directory "data_path".
    Reads PTB text files, converts strings to integer ids,
    and performs mini-batching of the inputs.
    The PTB dataset comes from Tomas Mikolov's webpage:
    http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
    Args:
        data_path: string path to the directory where simple-examples.tgz has
        been extracted.
    Returns:
        tuple (train_data, valid_data, test_data, vocabulary)
        where each of the data objects can be passed to PTBIterator.
    """
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    return train_data, valid_data, test_data, vocabulary


def ptb_iterator(raw_data, batch_size, num_steps):
    """
    Iterate on the raw PTB data.
    This generates batch_size pointers into the raw PTB data, and allows
    minibatch iteration along these pointers.
    Args:
        raw_data: one of the raw data outputs from ptb_raw_data.
        batch_size: int, the batch size.
        num_steps: int, the number of unrolls.
    Yields:
        Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
        The second element of the tuple is the same data time-shifted to the
        right by one.
    Raises:
        ValueError: if batch_size or num_steps are too high.
    """
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x, y)


def test_reader():
    """
    测试读取数据。
    :return:
    """
    train_path = os.path.join(ptb_data_path, "ptb.train.txt")
    result = _build_vocab(train_path)
    common_logger.info(result)


def test_ptb_raw_data():
    """
    测试读取ptb原始数据。
    :return:
    """
    tmp_dir = './data2'
    _string_data = "\n".join(
        [" hello there i am",
         " rain as day",
         " want some cheesy puffs ?"])
    for suffix in "train", "valid", "test":
        filename = os.path.join(tmp_dir, "ptb.{0}.txt".format(suffix))
        with tf.gfile.GFile(filename, "w") as fh:
            fh.write(_string_data)
    # Smoke test
    output = ptb_raw_data(tmp_dir)
    common_logger.info(len(output))


def test_ptb_iterator():
    """
    测试ptb生成器。
    :return:
    """
    raw_data = [4, 3, 2, 1, 0, 5, 6, 1, 1, 1, 1, 0, 3, 4, 1]
    batch_size = 3
    num_steps = 2
    x, y = ptb_iterator(raw_data, batch_size, num_steps)
    common_logger.info("x:{0},y:{1}".format(x, y))


if __name__ == '__main__':
    pass
    test_reader()
    # test_ptb_iterator()
    # test_ptb_raw_data()
