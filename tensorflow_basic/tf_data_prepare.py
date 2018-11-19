# -*- coding:utf-8 -*-

# ==============================================================================
# tensorflow 数据准备工作。
# ==============================================================================
import os
import struct
import numpy as np
from PIL import Image
import config.common_config as com_config
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()
mnist_dir = com_config.MNIST_DIR


def load_mnist(path, kind='train'):
    """
    加载MNIST数据，Load MNIST data from `path`
    """
    labels_path = os.path.join(path, '{0}-labels.idx1-ubyte'.format(kind))
    images_path = os.path.join(path, '{0}-images.idx3-ubyte'.format(kind))

    with open(labels_path, 'rb') as label_file:
        magic, n = struct.unpack('>II', label_file.read(8))
        labels = np.fromfile(label_file, dtype=np.uint8)

    with open(images_path, 'rb') as image_file:
        magic, num, rows, cols = struct.unpack('>IIII', image_file.read(16))
        images = np.fromfile(image_file, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def test_show_image():
    """
    测试显示图像。
    :return:
    """
    image_arrays, label_arrays = load_mnist(mnist_dir, "train")
    common_logger.info("标签数组的shape:{0}".format(label_arrays.shape))
    common_logger.info("图像数组的shape:{0}".format(image_arrays.shape))
    common_logger.info("标签1的内容:{0}".format(label_arrays[0]))
    common_logger.info("图像1的内容:{0}".format(image_arrays[0]))
    image = Image.fromarray(image_arrays[0].reshape(28, 28))
    image.show()

if __name__ == "__main__":
    load_mnist(mnist_dir, "train")
    pass

