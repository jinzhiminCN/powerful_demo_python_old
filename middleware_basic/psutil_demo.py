# -*- coding:utf-8 -*-

# ==============================================================================
# 测试psutil的基本用法。
# ==============================================================================
import psutil
import sys
import traceback
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()
# 1G单位的byte大小
G_BYTE = 1024*1024*1024


def test_cpu():
    """
    测试cpu的相关信息。
    :return:
    """
    cpu_times = psutil.cpu_times()
    common_logger.info("cpu times:{0}".format(cpu_times))

    cpu_count = psutil.cpu_count()
    common_logger.info("cpu 逻辑个数:{0}".format(cpu_count))

    cpu_count = psutil.cpu_count(logical=False)
    common_logger.info("cpu 物理个数:{0}".format(cpu_count))

    cpu_percent = psutil.cpu_percent()
    common_logger.info("cpu 使用率:{0}".format(cpu_percent))

    for x in range(10):
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        common_logger.info("cpu percent: {0}".format(cpu_percent))


def test_memory():
    """
    测试内存的相关信息。
    :return:
    """
    mem = psutil.virtual_memory()
    common_logger.info("memory:{0}".format(mem))

    common_logger.info("memory total:{0}".format(mem.total/G_BYTE))
    common_logger.info("memory used:{0}".format(mem.used/G_BYTE))
    common_logger.info("memory free:{0}".format(mem.free/G_BYTE))
    common_logger.info("memory percent:{0}".format(mem.percent))


def test_disk():
    """
    测试磁盘的相关信息。
    :return:
    """
    # 磁盘分区信息
    disk_partitions = psutil.disk_partitions()
    common_logger.info("disk_partitions:{0}".format(disk_partitions))

    # 磁盘使用情况
    disk_usage = psutil.disk_usage('/')
    common_logger.info("disk_usage:{0}".format(disk_usage))

    # 磁盘IO
    disk_io_counters = psutil.disk_io_counters()
    common_logger.info("disk_io_counters:{0}".format(disk_io_counters))

    # 磁盘IO
    disk_io_counters = psutil.disk_io_counters(perdisk=True)
    common_logger.info("disk_io_counters:{0}".format(disk_io_counters))


def test_net():
    """
    测试磁盘的网络信息。
    :return:
    """
    # 获取网络总的io情况
    net_io_counters = psutil.net_io_counters()
    common_logger.info("net_io_counters:{0}".format(net_io_counters))

    net_io_counters = psutil.net_io_counters(pernic=True)
    common_logger.info("net_io_counters:{0}".format(net_io_counters))

    # 获取网络接口信息
    net_if_addrs = psutil.net_if_addrs()
    common_logger.info("net_if_addrs:{0}".format(net_if_addrs))

    # 获取网络接口状态
    net_if_stats = psutil.net_if_stats()
    common_logger.info("net_if_stats:{0}".format(net_if_stats))


def test_process():
    """
    测试进程信息。
    :return:
    """
    pids = psutil.pids()
    common_logger.info("pids:{0}".format(pids))

    p = psutil.Process(123164)

    common_logger.info("process name:{0}, exe:{1}, cwd:{2}"
                       .format(p.name(), p.exe(), p.cwd()))
    common_logger.info("process status:{0}, create_time:{1}, cpu_times:{2}"
                       .format(p.status(), p.create_time(), p.cpu_times()))
    common_logger.info("process files:".format(p.open_files()))

    try:
        common_logger.info("process uids:{0}, gids:{1}, num_threads:{2}"
                           .format(p.uids(), p.gids(), p.num_threads()))
    except AttributeError as error:
        common_logger.info("error:{0}".format(error))
        # common_logger.exception("error:{0}".format(error))
        # 指名输出栈踪迹
        # common_logger.error(error, exc_info=1)
        # 更加严重的错误级别
        # common_logger.critical(error, exc_info=1)
        traceback.print_exc()


if __name__ == "__main__":
    # test_cpu()
    # test_memory()
    # test_disk()
    # test_net()
    test_process()
    pass
