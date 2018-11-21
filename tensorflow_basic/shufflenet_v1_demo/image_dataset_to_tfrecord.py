# -*- coding:utf-8 -*-

# ==============================================================================
# 将完成分类的图像数据集生成tfrecord。
# ==============================================================================
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
import os
import io
from util.log_util import LoggerUtil
from tensorflow_basic.shufflenet_v1_demo import shufflenet_constant

# 日志器
common_logger = LoggerUtil.get_common_logger()

imagenet_dir = shufflenet_constant.TINY_IMAGENET_DIR
train_dir = os.path.join(imagenet_dir, "training")
val_dir = os.path.join(imagenet_dir, "validation")
tfrecord_dir = os.path.join(imagenet_dir, "tfrecord")
train_tfrecord_path = shufflenet_constant.TRAIN_TFRECORD_PATH
val_tfrecord_path = shufflenet_constant.VAL_TFRECORD_PATH


def _bytes_feature(value):
    """
    转换为字节类型的特征。
    :param value:
    :return:
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """
    转换为int64类型的特征。
    :param value:
    :return:
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# here you can also just use `return array.tostring()`
# but it will make tfrecords files a lot larger and
# you will need to change the input pipeline
def to_bytes(array):
    image = Image.fromarray(array)
    tmp = io.BytesIO()
    image.save(tmp, format='jpeg')
    return tmp.getvalue()


def create_encoder(folder):
    """Encode directories in the folder with integer values.
    Values are in the range 0..(n_directories - 1).

    Arguments:
        folder: A path to a folder where directories with images are.
            Each directory - separate class.
    Returns:
        A dict.
    """
    classes = os.listdir(folder)
    encoder = {n: i for i, n in enumerate(classes)}
    return encoder


def collect_metadata(folder, encoder):
    """Collect paths to images. Collect their classes.
    All paths must be with respect to 'folder'.

    Arguments:
        folder: A path to a folder where directories with images are.
            Each directory - separate class.
        encoder: A dict, folder_name -> integer.
    Returns:
        A pandas dataframe.
    """
    sub_dirs = list(os.walk(folder))[1:]
    metadata = []

    for dir_path, _, files in sub_dirs:
        dir_name = dir_path.split(os.path.sep)[-1]
        for file_name in files:
            image_metadata = [dir_name, os.path.join(dir_name, file_name)]
            metadata.append(image_metadata)

    m_dataframe = pd.DataFrame(metadata)
    m_dataframe.columns = ['class_name', 'img_path']

    # encode folder names by integers
    m_dataframe['class_number'] = m_dataframe.class_name.apply(lambda x: encoder[x])

    # shuffle the dataframe
    m_dataframe = m_dataframe.sample(frac=1).reset_index(drop=True)

    return m_dataframe


def convert(folder, encoder, tfrecords_filename):
    """Convert a folder with directories of images to tfrecords format.

    Arguments:
        folder: A path to a folder where directories with images are.
        encoder: A dict, folder_name -> integer.
        tfrecords_filename: A path where to save tfrecords file.
    """
    images_metadata = collect_metadata(folder, encoder)
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    for _, row in tqdm(images_metadata.iterrows()):
        file_path = os.path.join(folder, row.img_path)

        # read an image
        image = Image.open(file_path)

        # convert to an array
        array = np.asarray(image, dtype='uint8')

        # some images are grayscale
        if array.shape[-1] != 3:
            array = np.stack([array, array, array], axis=2)

        # get class of the image
        target = int(row.class_number)

        feature = {
            'image': _bytes_feature(to_bytes(array)),
            'target': _int64_feature(target),
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    writer.close()


def main():
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(tfrecord_dir, exist_ok=True)

    encoder = create_encoder(train_dir)
    # now you can get a folder's name from a class index

    np.save(os.path.join(tfrecord_dir, 'class_encoder.npy'), encoder)
    convert(train_dir, encoder, train_tfrecord_path)
    convert(val_dir, encoder, val_tfrecord_path)

    common_logger.info('Created two tfrecords files:')
    common_logger.info(train_tfrecord_path)
    common_logger.info(val_tfrecord_path)


if __name__ == "__main__":
    main()
    pass
