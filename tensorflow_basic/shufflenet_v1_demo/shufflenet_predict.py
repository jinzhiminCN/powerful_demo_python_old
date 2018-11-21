# -*- coding:utf-8 -*-

# ==============================================================================
# shufflenet神经网络模型的预测。
# ==============================================================================
import numpy as np
import pandas as pd
import json
import tensorflow as tf
from PIL import Image
import os
from util.log_util import LoggerUtil
from tensorflow_basic.shufflenet_v1_demo import shufflenet_constant
from tensorflow_basic.shufflenet_v1_demo.shufflenet_model import get_shufflenet

# 日志器
common_logger = LoggerUtil.get_common_logger()

imagenet_dir = shufflenet_constant.TINY_IMAGENET_DIR

log_dir = os.path.join(imagenet_dir, 'log')
save_dir = os.path.join(imagenet_dir, 'saved')
# folder where validation dataset is
validation_dir = os.path.join(imagenet_dir, 'validation')
# this file is created when you run `image_dataset_to_tfrecords.py`
class_encoder_path = os.path.join(imagenet_dir, 'class_encoder.npy')
# this file comes with dataset
class_names_path = os.path.join(imagenet_dir, 'words.txt')


def predict_prob(graph, ops, x_data, run):
    """Predict probabilities with a fitted model.

    Arguments:
        graph: A Tensorflow graph.
        ops: A dict of ops of the graph.
        x_data: A numpy array of shape [n_samples, image_size, image_size, 3]
            and of type 'float32', a batch of images with
            pixel values in range [0, 1].
        run: An integer that determines a folder where a fitted model
            is saved.

    Returns:
        predictions: A numpy array of shape [n_samples, n_classes]
            and of type 'float32'.
    """
    sess = tf.Session(graph=graph)
    ops['saver'].restore(sess, os.path.join('saved', 'model'))

    feed_dict = {'inputs/X:0': x_data, 'control/is_training:0': False}
    predictions = sess.run(ops['predictions'], feed_dict)

    sess.close()
    return predictions


def test_predict():
    """

    :return:
    """
    groups = 3
    complexity_scale_factor = 0.75

    graph, ops = get_shufflenet(
        groups=groups,
        complexity_scale_factor=complexity_scale_factor
    )

    filenames = ['n02056570/val_1073.JPEG', 'n02106662/val_3641.JPEG', 'n01774384/val_1684.JPEG']
    images = [Image.open(validation_dir + name) for name in filenames]

    images = [image.resize((56, 56)) for image in images]
    x_data = [np.asarray(image).astype('float32') for image in images]
    # batch
    x_data = np.stack(x_data)
    # normalize to [0, 1] range
    x_data /= 255.0

    predictions = predict_prob(graph, ops, x_data)

    # folder name -> class name in human readable format
    class_names = pd.read_csv(class_names_path, sep='\t', header=None)
    names = dict(class_names.set_index(0)[1])

    # folder name -> class index
    encoder = np.load(class_encoder_path)[()]

    # class index -> class name in human readable format
    decoder = {encoder[i]: names[i] for i in encoder}

    common_logger.info([decoder[i] for i in predictions.argmax(1)])


if __name__ == "__main__":
    pass