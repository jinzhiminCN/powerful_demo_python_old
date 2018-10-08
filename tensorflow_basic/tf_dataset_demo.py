# -*- coding:utf-8 -*-

# ==============================================================================
# tensorflow中tfrecord的简单使用。将图片数据生成tfrecord格式，并用来训练mnist数据。
# ==============================================================================
import math
from functools import partial
from tensorflow_basic.base_dnn_tf import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
mnist_dir = com_config.MNIST_DIR


def get_tf_dataset(dataset_text_file, batch_size=64, channels=3, crop_size=[28, 28],
                   shuffle_size=200, augmentation=False):
    def aug_1(image):
        image = tf.image.random_brightness(image, max_delta=2./255.)
        image = tf.image.random_saturation(image, lower=0.01, upper=0.05)
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.01, upper=0.05)
        return image

    def aug_2(image):
        image = tf.image.random_saturation(image, lower=0.01, upper=0.05)
        image = tf.image.random_brightness(image, max_delta=2./255.)
        image = tf.image.random_contrast(image, lower=0.01, upper=0.05)
        image = tf.image.random_hue(image, max_delta=0.05)
        return image

    def aug_3(image):
        image = tf.image.random_contrast(image, lower=0.01, upper=0.05)
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_brightness(image, max_delta=2./255.)
        image = tf.image.random_saturation(image, lower=0.01, upper=0.05)
        return image

    def aug_4(image):
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_saturation(image, lower=0.01, upper=0.05)
        image = tf.image.random_contrast(image, lower=0.01, upper=0.05)
        image = tf.image.random_brightness(image, max_delta=2./255.)
        return image

    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=channels)
        image = tf.image.resize_images(image_decoded, crop_size)
        image = tf.cast(image, tf.float32) * (1./255) - 0.5
        if augmentation:
            # tensorflow1.7支持tf.contrib.image.rotate
            angle = tf.reshape(tf.random_uniform([1], -math.pi / 12, math.pi / 12, tf.float32), [])
            image = tf.contrib.image.rotate(image, angle)
            # image = tf.image.random_flip_left_right(image)
            # image = tf.random_crop(image, [crop_size, crop_size, 3])

            p1 = partial(aug_1, image)
            p2 = partial(aug_2, image)
            p3 = partial(aug_3, image)
            p4 = partial(aug_4, image)

            k = tf.reshape(tf.random_uniform([1], 0, 4, tf.int32), [])
            image = tf.case([(tf.equal(k, 0), p1),
                             (tf.equal(k, 1), p2),
                             (tf.equal(k, 2), p3),
                             (tf.equal(k, 3), p4)],
                            default=p1,
                            exclusive=True)

        return image, label

    def read_labeled_image_list(dataset_text_file):
        filenames = []
        labels = []

        with open(dataset_text_file, "r", encoding="utf-8") as f_l:
            filenames_lables = f_l.readlines()

        for filename_lable in filenames_lables:
            filename_array = filename_lable.split(" ")
            filenames.append(filename_array[0])
            labels.append(int(filename_array[1].strip("\n")))
        return filenames, labels

    def read_img_file_label_list(dir_path):
        """
        读取图像文件名和标签的列表。
        :return:
        """
        img_labels = []
        img_paths = []

        img_dir = "{0}images".format(dir_path[:-6])
        for filename in os.listdir(dir_path):
            label_path = os.path.join(dir_path, filename)
            img_filename = "{0}png".format(filename[:-3])
            img_path = os.path.join(img_dir, img_filename)
            img_paths.append(img_path)
            with open(label_path, mode="r", encoding="utf-8") as file:
                label = file.read()
                label = label.strip()
                img_labels.append(int(label))

        return img_paths, img_labels

    filenames, labels = read_img_file_label_list(dataset_text_file)
    # filenames, labels = read_labeled_image_list(dataset_text_file)

    filenames = tf.constant(filenames, name='filename_list')
    labels = tf.constant(labels, name='label_list')

    # tensorflow1.3:tf.contrib.data.Dataset.from_tensor_slices
    # tensorflow1.4+:tf.data.Dataset.from_tensor_slices
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.shuffle(shuffle_size)
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.repeat()

    return dataset


def train():
    """
    训练神经网络。
    :param
    :return:
    """
    batch_size = 1024

    # 模型输入
    x_inputs = tf.placeholder(tf.float32, [batch_size, 28, 28, 3], name='inputs')

    # 网络结构
    conv1 = tf.layers.conv2d(inputs=x_inputs, filters=64, kernel_size=(3, 3), padding="same", activation=None)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=(3, 3), padding="same", activation=None)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 128])
    fc1 = tf.layers.dense(pool2_flat, 500, activation=tf.nn.relu)
    fc2 = tf.layers.dense(fc1, 10, activation=tf.nn.relu)
    y_pred = tf.nn.softmax(fc2)

    # 模型标签
    y_label = tf.placeholder(tf.float32, [batch_size, 10])
    # 计算交叉熵
    cross_entropy = -tf.reduce_mean(y_label * tf.log(y_pred))
    # 学习率
    learning_rate = 1e-3
    # 优化器
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    # 判断预测标签和实际标签是否匹配
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # 批量获取数据
    dir_path = os.path.join(mnist_dir, "train.labels")
    dataset = get_tf_dataset(dataset_text_file=dir_path, batch_size=batch_size)
    # dataset = get_tf_dataset(dataset_text_file="./train.txt", batch_size=batch_size)
    iterator = dataset.make_one_shot_iterator()
    img_batch, label_batch = iterator.get_next()

    init = tf.global_variables_initializer()

    # 执行训练会话
    with tf.Session() as session:
        session.run(init)
        threads = tf.train.start_queue_runners()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
        for i in range(1000):
            tmp_batch_img, tmp_batch_label = session.run([img_batch, tf.one_hot(label_batch, depth=10)])
            feed = {x_inputs: tmp_batch_img, y_label: tmp_batch_label}
            _, loss, acc = session.run([train_step, cross_entropy, accuracy], feed_dict=feed)

            common_logger.info("step:{0} loss:{1:>7} accuracy:{2:>7}".format(i, loss, acc))
        saver.save(session, "./save/mnist.ckpt")


if __name__ == "__main__":
    train()
