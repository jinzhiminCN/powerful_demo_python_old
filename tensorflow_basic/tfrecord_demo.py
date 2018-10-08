# -*- coding:utf-8 -*-

# ==============================================================================
# tensorflow中tfrecord的简单使用。将图片数据生成tfrecord格式，并用来训练mnist数据。
# ==============================================================================
from PIL import Image
from tensorflow_basic.base_dnn_tf import *

# 可设置环境变量CUDA_VISIBLE_DEVICES，指明可见的cuda设备
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
mnist_dir = com_config.MNIST_DIR


def data_to_tfrecord(image_paths, labels, filename):
    """
    Save data into TFRecord.
    :param image_paths:
    :param labels:
    :param filename:输出的tfrecord文件名称
    :return:
    """
    if os.path.isfile(filename):
        common_logger.info("{0} exists".format(filename))
        return

    common_logger.info("Converting data into {0} ...".format(filename))

    # 创建TFRecordWriter
    writer = tf.python_io.TFRecordWriter(filename)
    for index, img_path in enumerate(image_paths):
        common_logger.info("{0}".format(index))
        # image
        img = Image.open(img_path)
        img = img.resize((28, 28))
        img_raw = img.tobytes()
        # label
        label = int(labels[index])
        # example
        example = make_example(label, img_raw)

        # Serialize To String
        writer.write(example.SerializeToString())
    writer.close()


def make_example(label, img_raw):
    """
    构造tfrecord中的Example。
    :param label:图像标签
    :param img_raw:原始图像
    :return:
    """
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "label": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[label])
                ),
                "img_raw": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[img_raw])
                ),
            }))

    return example


def tfrecord_to_data(filename):
    """
    generate a queue with a given file name.
    :param filename:
    :return:
    """
    common_logger.info("reading tfrecord from {0}".format(filename))

    # 文件名队列
    filename_queue = tf.train.string_input_producer([filename])
    # 创建TFRecordReader
    reader = tf.TFRecordReader()
    # 获取下一个record(key, value)
    _, serialized_example = reader.read(filename_queue)
    # 解析example数据
    features = tf.parse_single_example(serialized_example, features={
        # 固定长度label
        'label': tf.FixedLenFeature([], tf.int64),
        # 可变长度label
        # 'label': tf.VarLenFeature(tf.int64),
        'img_raw': tf.FixedLenFeature([], tf.string),
    })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [28, 28, 1])
    img = tf.cast(img, tf.float32) * (1./255) - 0.5
    label = tf.cast(features['label'], tf.int64)
    return img, label


def decode_tfrecords(filename):
    """
    show the tfrecords.
    :param filename:
    :return:
    """
    for serialized_example in tf.python_io.tf_record_iterator(filename):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)

        image = example.features.feature['img_raw'].bytes_list.value
        label = example.features.feature['label'].int64_list.value
        common_logger.info("image: {0}, label: {1}".format(image, label))


def read_data_from_file(dir_path, filename):
    """
    从文件中读取数据。文件中存放了图像文件名和图像标签。
    :param dir_path:
    :param filename:
    :return:
    """
    img_labels = []
    img_paths = []

    file_path = os.path.join(dir_path, filename)
    train_txt = open(file_path, 'r')

    for idx in train_txt:
        idx = idx.rstrip('\n')
        spt = idx.split(' ')
        img_paths.append(os.path.join(dir_path, spt[0]))
        img_labels.append(spt[1])

    return img_paths, img_labels


def read_data_from_dir(dir_path):
    """
    从标签目录中读取图像文件名和图像标签。
    :param dir_path:
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
            img_labels.append(label)

    return img_paths, img_labels


def train(tfrecord_path):
    """
    训练神经网络。
    :param tfrecord_path
    :return:
    """
    batch_size = 1000

    # 模型输入
    x_inputs = tf.placeholder(tf.float32, [batch_size, 28, 28, 1], name='inputs')

    # 构造网络结构
    conv1 = tf.layers.conv2d(inputs=x_inputs, filters=64, kernel_size=(3, 3), padding="same", activation=None)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=(3, 3), padding="same", activation=None)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 128])
    full_conn1 = tf.layers.dense(pool2_flat, 1000, activation=tf.nn.relu)
    y_output = tf.layers.dense(full_conn1, 10, activation=tf.nn.relu)
    y_predict = tf.nn.softmax(y_output)

    # 模型标签
    y_label = tf.placeholder(tf.float32, [batch_size, 10])
    # 计算交叉熵
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_output, labels=y_label))
    # cross_entropy = -tf.reduce_mean(y_label * tf.log(y_predict))
    # 学习率
    learning_rate = 1e-3
    # 优化器
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    # 判断预测标签和实际标签是否匹配
    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    img, label = tfrecord_to_data(tfrecord_path)
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=batch_size,
                                                    capacity=2000,
                                                    min_after_dequeue=1000)

    init = tf.global_variables_initializer()

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


def test_tfrecord():
    """
    测试tfrecord相关的内容。
    :return:
    """
    data_type = "train"
    # 1. 读取原始数据
    dir_path = os.path.join(mnist_dir, "{0}.labels".format(data_type))
    img_paths, img_labels = read_data_from_dir(dir_path)
    # for i in range(10):
    #     common_logger.info("image path:{0}, image label:{1}".format(img_paths[i], img_labels[i]))
    tfrecord_name = "mnist_{0}.tfrecords".format(data_type)
    tfrecord_path = os.path.join(mnist_dir, tfrecord_name)
    # 2. 生成tfrecord文件
    data_to_tfrecord(img_paths, img_labels, tfrecord_path)
    # 3. 解码tfrecord的内容
    # decode_tfrecords(tfrecord_path)
    # 4. 从tfrecord中获取数据
    # img, label = tfrecord_to_data(tfrecord_path)
    # common_logger.info("img:{0}, label:{1}".format(img, label))


def test_train():
    """
    测试使用tfrecord进行训练的方法。
    :return:
    """
    tfrecord_name = "mnist_train.tfrecords"
    tfrecord_path = os.path.join(mnist_dir, tfrecord_name)
    train(tfrecord_path)


if __name__ == "__main__":
    pass
    # test_tfrecord()
    test_train()
