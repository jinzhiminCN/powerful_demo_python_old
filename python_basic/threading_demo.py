# -*- coding:utf-8 -*-

# ==============================================================================
# 测试threading的相关方法。
# https://www.cnblogs.com/tkqasn/p/5700281.html
# ==============================================================================
import time
import threading
import multiprocessing
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()


def loop_action():
    """
    线程执行代码。
    :return:
    """
    common_logger.info('thread {0} is running...'.format(threading.current_thread().name))
    n = 0
    while n < 5:
        n += 1
        common_logger.info('thread {0} >>> {1}'.format(threading.current_thread().name, n))
        time.sleep(1)
    common_logger.info('thread {0} ended.'.format(threading.current_thread().name))


def loop_arg_action(arg):
    """
    包含参数的线程执行代码。
    :param arg:
    :return:
    """
    time.sleep(1)
    common_logger.info('sub thread start! the thread name is:{}\r'
                       .format(threading.currentThread().getName()))
    common_logger.info('the arg is:{0}\r'.format(arg))


def test_multi_cpu_threading():
    """
    测试多个cpu的多线程运行。
    启动与CPU核心数量相同的N个线程，在4核CPU上可以监控到CPU占用率仅有102%，也就是仅使用了一核。
    但是用C、C++或Java来改写相同的死循环，直接可以把全部核心跑满，4核就跑到400%，8核就跑到800%，为什么Python不行呢？
    因为Python的线程虽然是真正的线程，但解释器执行代码时，有一个GIL锁：Global Interpreter Lock，
    任何Python线程执行前，必须先获得GIL锁，然后，每执行100条字节码，解释器就自动释放GIL锁，
    让别的线程有机会执行。这个GIL全局锁实际上把所有线程的执行代码都给上了锁，所以，多线程在Python中只能交替执行，
    即使100个线程跑在100核CPU上，也只能用到1个核。
    GIL是Python解释器设计的历史遗留问题，通常我们用的解释器是官方实现的CPython，要真正利用多核，除非重写一个不带GIL的解释器。
    所以，在Python中，可以使用多线程，但不要指望能有效利用多核。如果一定要通过多线程利用多核，那只能通过C扩展来实现，
    不过这样就失去了Python简单易用的特点。不过，也不用过于担心，Python虽然不能利用多线程实现多核任务，
    但可以通过多进程实现多核任务。多个Python进程有各自独立的GIL锁，互不影响。
    :return:
    """
    for i in range(multiprocessing.cpu_count()):
        thd = threading.Thread(target=loop_action)
        thd.start()


class MyThread(threading.Thread):
    def __init__(self, arg):
        """
        初始化线程。
        :param arg:
        """
        # 注意：一定要显式的调用父类的初始化函数。
        super(MyThread, self).__init__()
        self.arg = arg

    def run(self):
        """
        定义每个线程要运行的函数。
        :return:
        """
        time.sleep(1)
        common_logger.info('the arg is:{0}\r'.format(self.arg))


def test_thread_start():
    """
    测试线程的start运行方法。
    :return:
    """
    thd = MyThread("test")
    thd.start()

    for i in range(0, 4):
        thd = threading.Thread(target=loop_arg_action, args=(i,))
        thd.setDaemon(False)
        thd.start()

    for i in range(6, 8):
        thd = threading.Thread(target=loop_arg_action, args=(i,))
        # 设置线程为后台线程
        thd.setDaemon(True)
        thd.start()

    common_logger.info("main_thread end!")


def test_thread_join():
    """
    测试线程的join阻塞方法。
    :return:
    """
    # 程序只能顺序执行，每个线程都被上一个线程的join阻塞，使得“多线程”失去了意义。
    start_time = time.time()
    for i in range(0, 4):
        thd = threading.Thread(target=loop_arg_action, args=(i,))
        thd.setDaemon(False)
        thd.start()
        thd.join()
    end_time = time.time()
    common_logger.info("持续时长：{0}".format(end_time - start_time))

    # 每个线程都先执行，然后阻塞主线程，主线程等待子线程全部执行完或者子线程超时后，主线程才结束。
    start_time = time.time()
    # 线程存放列表
    thread_list = []
    for i in range(4):
        t = threading.Thread(target=loop_arg_action, args=(i,))
        t.setDaemon(True)
        thread_list.append(t)

    for t in thread_list:
        t.start()

    for t in thread_list:
        t.join()

    end_time = time.time()
    common_logger.info("持续时长：{0}".format(end_time - start_time))


if __name__ == "__main__":
    # loop_action()
    # test_multi_cpu_threading()
    # test_thread_start()
    test_thread_join()
    pass
