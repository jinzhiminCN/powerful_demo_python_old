# -*- coding:utf-8 -*-

# ==============================================================================
# 移动ImageNet目录下的文件。这里是tiny-imagenet-200训练文件。
# ==============================================================================
import pandas as pd
import os
import shutil
from tqdm import tqdm
from util.log_util import LoggerUtil
from tensorflow_basic.shufflenet_v1_demo import shufflenet_constant

# 日志器
common_logger = LoggerUtil.get_common_logger()


# tiny-imagenet-200 的目录
tiny_imagenet_dir = shufflenet_constant.TINY_IMAGENET_DIR

data_dir = tiny_imagenet_dir


def move_image_data():
    """
    移动和拆分图像数据。
    :return:
    """
    # load validation metadata
    annotations_file = os.path.join(data_dir, 'val', 'val_annotations.txt')
    val_data = pd.read_csv(annotations_file, sep='\t', header=None)
    # drop bounding boxes info
    val_data.drop([2, 3, 4, 5], axis=1, inplace=True)
    val_data.columns = ['img_name', 'img_class']
    unique_classes = val_data.img_class.unique()

    # 移动验证数据
    move_valid_data(unique_classes, val_data)
    # 移动训练数据
    move_train_data(unique_classes)


def move_valid_data(unique_classes, val_data):
    """
    拆分和移动验证数据集。
    将目录val下的图像和标注文本生成验证数据，
    也就是将图像分类存放在目录validation下。
    :param: unique_classes
    :param: val_data
    :return:
    """
    common_logger.info('moving validation data')

    # create new folders to move the data into
    validation_dir = os.path.join(data_dir, 'validation')
    os.mkdir(validation_dir)
    for name in unique_classes:
        os.mkdir(os.path.join(validation_dir, name))

    # loop over all classes
    for name in tqdm(unique_classes):
        # choose images only from a specific class
        class_images = val_data.loc[val_data.img_class == name, 'img_name'].values
        # copy these images to a new folder
        for img in class_images:
            shutil.copyfile(
                os.path.join(data_dir, 'val', 'images', img),
                os.path.join(validation_dir, name, img)
            )

    common_logger.info('validation data is in {0}'.format(validation_dir))


def move_train_data(unique_classes):
    """
    拆分和移动验证数据集。
    :param: unique_classes
    :return:
    """
    common_logger.info('moving training data')

    # create new folders to move data into
    training_dir = os.path.join(data_dir, 'training')
    os.mkdir(training_dir)
    for name in unique_classes:
        os.mkdir(os.path.join(training_dir, name))

    # loop over all classes
    for name in tqdm(unique_classes):
        # choose images only from a specific class
        class_images = os.listdir(os.path.join(data_dir, 'train', name, 'images'))
        # copy these images to a new folder
        for img in class_images:
            shutil.copyfile(
                os.path.join(data_dir, 'train', name, 'images', img),
                os.path.join(training_dir, name, img)
            )

    common_logger.info('training data is in {0}'.format(training_dir))


if __name__ == "__main__":
    move_image_data()
    pass
