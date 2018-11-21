# -*- coding:utf-8 -*-

# ==============================================================================
# shufflenet网络的训练过程。
# ==============================================================================
import os
import sys
import time
from tqdm import tqdm
import shutil
import tensorflow as tf
from util.log_util import LoggerUtil
from tensorflow_basic.shufflenet_v1_demo import shufflenet_constant
from tensorflow_basic.shufflenet_v1_demo.shufflenet_model import get_shufflenet

# 日志器
common_logger = LoggerUtil.get_common_logger()

imagenet_dir = shufflenet_constant.TINY_IMAGENET_DIR

log_dir = os.path.join(imagenet_dir, 'log')
save_dir = os.path.join(imagenet_dir, 'saved')
tfrecord_dir = os.path.join(imagenet_dir, "tfrecord")

training_info_file = os.path.join(log_dir, 'training_info.txt')
model_config_file = os.path.join(log_dir, 'model_config.txt')
train_tfrecords = os.path.join(tfrecord_dir, 'train.tfrecords')
val_tfrecords = os.path.join(tfrecord_dir, 'val.tfrecords')

reset = True


def make_dirs():
    """
    构造相关目录。
    """
    if reset and os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    if reset and os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


def train_model():
    """
    训练模型。
    """
    # 初始学习率
    initial_lr = 1e-1
    batch_size = 200
    num_epochs = 35
    steps_per_epoch = 500
    validation_steps = 50
    patience = 10
    threshold = 0.01
    lr_threshold = 0.01
    lr_patience = 4
    weight_decay = 5e-3
    groups = 3
    dropout = 0.5
    complexity_scale_factor = 0.75

    # create the graph and start a session
    graph, ops = get_shufflenet(
        initial_lr, weight_decay,
        groups, dropout,
        complexity_scale_factor
    )

    sess = tf.Session(graph=graph)
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    common_logger.info('Created the graph and started a session!')

    # check if to continue training or start from scratch
    # warm start
    warm = os.path.exists(training_info_file)
    if warm and not reset:
        common_logger.info('Restoring previously saved model and continuing training.')
        initial_epoch = sum(1 for line in open(training_info_file))
        try:
            ops['saver'].restore(sess, os.path.join(save_dir, 'model'))
        except:
            common_logger.info('Can\'t restore the saved model, '
                               'maybe architectures don\'t match.')
            sys.exit()
    else:
        common_logger.info('Training model from scratch.')
        initial_epoch = 1
        sess.run(ops['init_variables'])

    # initialize data sources
    data_dict = {
        'input_pipeline/train_file:0': train_tfrecords,
        'input_pipeline/val_file:0': val_tfrecords,
        'input_pipeline/batch_size:0': batch_size
    }
    sess.run(ops['init_data'], data_dict)

    # training info will be collected here
    losses = []
    training_epochs = range(
        initial_epoch,
        initial_epoch + num_epochs
    )

    try:
        for epoch in training_epochs:
            start_time = time.time()
            running_loss, running_accuracy = 0.0, 0.0
            # 迭代器初始化训练数据集
            sess.run(ops['train_init'])

            # at zeroth step also collect metadata and summaries
            run_options = tf.RunOptions(
                trace_level=tf.RunOptions.FULL_TRACE
            )
            run_metadata = tf.RunMetadata()

            # do epoch's zeroth step
            _, batch_loss, batch_accuracy, summary, grad_summary = sess.run([
                ops['optimize'], ops['log_loss'], ops['accuracy'],
                ops['summaries'], ops['grad_summaries']
            ], options=run_options, run_metadata=run_metadata)
            running_loss += batch_loss
            running_accuracy += batch_accuracy

            common_logger.info('epoch {}'.format(epoch))
            training_steps = tqdm(
                range(1, steps_per_epoch),
                initial=1, total=steps_per_epoch
            )

            # main training loop
            for step in training_steps:
                _, batch_loss, batch_accuracy = sess.run([
                    ops['optimize'], ops['log_loss'], ops['accuracy']
                ])
                running_loss += batch_loss
                running_accuracy += batch_accuracy

            # evaluate on the validation set
            val_loss, val_accuracy = _evaluate(
                sess, ops, validation_steps
            )
            train_loss = running_loss/steps_per_epoch
            train_accuracy = running_accuracy/steps_per_epoch

            # collect all losses and accuracies
            losses += [(
                epoch, train_loss, val_loss,
                train_accuracy, val_accuracy, time.time() - start_time
            )]
            writer.add_run_metadata(run_metadata, str(epoch))
            writer.add_summary(summary, epoch)
            writer.add_summary(grad_summary, epoch)
            common_logger.info('loss: {0:.3f}, val_loss: {1:.3f}, '
                               'acc: {2:.3f}, val_acc: {3:.3f}, time: {4:.3f}'
                               .format(*losses[-1][1:]))

            # consider a possibility of early stopping
            if _is_early_stopping(losses, patience, threshold):
                common_logger.info('Early stopping!')
                break

            # consider a possibility of reducing learning rate by some factor
            _reduce_lr_on_plateau(
                sess, ops, losses,
                lr_patience, lr_threshold
            )
    except (KeyboardInterrupt, SystemExit):
        common_logger.info('Interruption detected, exiting the program...')

    common_logger.info('Writing logs and saving the trained model.')
    _write_training_info(
        losses, warm,
        training_info_file, model_config_file
    )
    ops['saver'].save(sess, os.path.join(save_dir, 'model'))
    sess.close()


def _evaluate(sess, ops, validation_steps):
    """
    评估损失函数。
    :param sess:
    :param ops:
    :param validation_steps:
    :return:
    """
    val_loss, val_accuracy = 0.0, 0.0
    sess.run(ops['val_init'])

    for i in range(validation_steps):
        batch_loss, batch_accuracy = sess.run(
            [ops['log_loss'], ops['accuracy']],
            {'control/is_training:0': False}
        )
        val_loss += batch_loss
        val_accuracy += batch_accuracy

    val_loss /= validation_steps
    val_accuracy /= validation_steps
    return val_loss, val_accuracy


def _is_early_stopping(losses, patience=10, threshold=0.01):
    """
    是否提前结束。
    it decides if training must stop.
    :param losses:
    :param patience:
    :param threshold:
    :return:
    """
    # get validation set accuracies
    accuracies = [x[4] for x in losses]

    if len(losses) > (patience + 4):
        # running average
        average = (accuracies[-(patience + 4)] +
                   accuracies[-(patience + 3)] +
                   accuracies[-(patience + 2)] +
                   accuracies[-(patience + 1)] +
                   accuracies[-patience])/5.0
        return accuracies[-1] < average + threshold
    else:
        return False


def _reduce_lr_on_plateau(
        sess, ops, losses,
        patience=10, threshold=0.01):

    # get validation set accuracies
    accuracies = [x[4] for x in losses]

    if len(losses) > (patience + 4):
        # running average
        average = (accuracies[-(patience + 4)] +
                   accuracies[-(patience + 3)] +
                   accuracies[-(patience + 2)] +
                   accuracies[-(patience + 1)] +
                   accuracies[-patience])/5.0
        if accuracies[-1] < (average + threshold):
            sess.run(ops['drop_learning_rate'])
            print('Learning rate is dropped!\n')


def _write_training_info(
        losses, warm,
        training_info_file, model_config_file):

    mode = 'a' if warm else 'w'
    with open(training_info_file, mode) as f:

        # if file is new then add columns
        if not warm:
            columns = ('epoch,train_loss,val_loss,'
                       'train_accuracy,val_accuracy,time\n')
            f.write(columns)

        for i in losses:
            values = ('{0},{1:.3f},{2:.3f},'
                      '{3:.3f},{4:.3f},{5:.3f}\n').format(*i)
            f.write(values)


def main():
    """
    主运行类。
    :return:
    """
    make_dirs()
    train_model()


if __name__ == "__main__":
    main()
    pass

