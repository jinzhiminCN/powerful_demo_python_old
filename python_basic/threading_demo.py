# -*- coding:utf-8 -*-

# ==============================================================================
# 测试threading的相关方法。
# https://www.cnblogs.com/tkqasn/p/5700281.html
#     Lock（指令锁）: Lock是可用的最低级的同步指令。Lock处于锁定状态时，不被特定的线程拥有。Lock
# 包含两种状态——锁定和非锁定，以及两个基本的方法。可以认为Lock有一个锁定池，当线程请求锁时，将线程置
# 于池中，直到获得锁后出池。池中的线程处于状态图中的同步阻塞状态。
#     RLock（可重入锁）: RLock是一个可以被同一个线程请求多次的同步指令。RLock使用了“拥有的线程”和
# “递归等级”的概念，处于锁定状态时，RLock被某个线程拥有。拥有RLock的线程可以再次调用acquire()，释
# 放锁时需要调用release()相同次数。可以认为RLock包含一个锁定池和一个初始值为0的计数器，每次成功调用
# acquire()/release()，计数器将+1/-1，为0时锁处于未锁定状态。
#     Condition（条件变量）：Condition通常与一个锁关联。需要在多个Condition中共享一个锁时，可以传
# 递一个Lock/RLock实例给构造方法，否则它将自己生成一个RLock实例。可以认为，除了Lock带有的锁定池外，
# Condition还包含一个等待池，池中的线程处于等待阻塞状态，直到另一个线程调用notify()/notifyAll()通
# 知；得到通知后线程进入锁定池等待锁定。
# ==============================================================================
import time
import threading
import multiprocessing
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()

# threading.local是一个小写字母开头的类，用于管理 thread-local（线程局部的）数据。
# 对于同一个local，线程无法访问其他线程设置的属性；线程设置的属性不会被其他线程设置的同名属性替换。
# 可以把local看成是一个“线程-属性字典”的字典，local封装了从自身使用线程作为key检索对应的属性字典、
# 再使用属性名作为key检索属性值的细节。
# 设置线程局部变量
local_value = threading.local()
# 账号余额
balance = 0
# 使用锁
lock = threading.Lock()

# 商品
product = None
# 条件变量
con = threading.Condition()
condition = threading.Condition()
# 商品数量
products = 0
product_list = None


def change_it(n):
    """
    修改balance的值。
    :param n:
    :return:
    """
    # 先存后取，结果应该为0:
    global balance
    balance = balance + n
    time.sleep(0.5)
    balance = balance - n
    common_logger.info('current balance: {0}'.format(balance))


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


def print_local_name():
    """
    获取当前线程关联的name。
    :return:
    """
    name = local_value.name
    common_logger.info('Hello, {0} (in {1})'
                       .format(name, threading.current_thread().name))


def show_thread_name(name):
    """
    绑定ThreadLocal的name。
    :param name:
    :return:
    """
    local_value.name = name
    print_local_name()


def test_thread_local():
    """
    测试threading.local类的使用。
    :return:
    """
    t1 = threading.Thread(target=show_thread_name, args=('Alice',), name='Thread-A')
    t2 = threading.Thread(target=show_thread_name, args=('Bob',), name='Thread-B')
    t1.start()
    t2.start()
    t1.join()
    t2.join()


def change_balance(val):
    """
    多次修改balance。
    :param val:
    :return:
    """
    for i in range(1000):
        change_it(val)


def change_lock_balance(val):
    """
    多次修改带锁的balance。
    :param val:
    :return:
    """
    for i in range(100000):
        # 先要获取锁:
        lock.acquire()
        try:
            change_it(val)
        finally:
            # 使用完要释放锁:
            lock.release()


def test_balance():
    """
    测试修改balance。
    :return:
    """
    t1 = threading.Thread(target=change_balance, args=(5,))
    t2 = threading.Thread(target=change_balance, args=(8,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    common_logger.info('finally balance: '.format(balance))


def test_balance_with_lock():
    """
    测试带锁地修改balance。
    :return:
    """
    t1 = threading.Thread(target=change_lock_balance, args=(5,))
    t2 = threading.Thread(target=change_lock_balance, args=(8,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    common_logger.info('finally balance: '.format(balance))


def produce():
    """
    生产者方法
    :return:
    """

    global product

    if con.acquire():
        while True:
            if product is None:
                common_logger.info('produce...')
                product = 'anything'

                # 通知消费者，商品已经生产
                con.notify()

            # 等待通知
            con.wait()
            time.sleep(2)


def consume():
    """
    消费者方法
    :return:
    """

    global product

    if con.acquire():
        while True:
            if product is not None:
                common_logger.info('consume...')
                product = None

                # 通知生产者，商品已经没了
                con.notify()

            # 等待通知
            con.wait()
            time.sleep(2)


def test_condition():
    """
    测试条件变量。
    :return:
    """
    t1 = threading.Thread(target=produce)
    t2 = threading.Thread(target=consume)
    t2.start()
    t1.start()


class Producer(threading.Thread):
    def run(self):
        global products
        while True:
            if condition.acquire():
                if products < 10:
                    products += 1
                    common_logger.info("Producer({0}):deliver one, now products:{1}"
                                       .format(self.name, products))
                    # 不释放锁定，因此需要下面一句
                    condition.notify()
                    condition.release()
                else:
                    common_logger.info("Producer({0}):already 10, stop deliver, now products:{1}"
                                       .format(self.name, products))
                    # 自动释放锁定
                    condition.wait()
                time.sleep(2)


class Consumer(threading.Thread):
    def run(self):
        global products
        while True:
            if condition.acquire():
                if products > 1:
                    products -= 1
                    common_logger.info("Consumer({0}):consume one, now products:{1}"
                                       .format(self.name, products))
                    condition.notify()
                    condition.release()
                else:
                    common_logger.info("Consumer({0}):only 1, stop consume, products:{1}"
                                       .format(self.name, products))
                    condition.wait()
                time.sleep(2)


def test_producer_consumer():
    """

    :return:
    """
    for p in range(0, 2):
        p = Producer()
        p.start()

    for c in range(0, 3):
        c = Consumer()
        c.start()


def do_set():
    if condition.acquire():
        while product_list is None:
            condition.wait()
        for i in range(len(product_list))[::-1]:
            product_list[i] = 1
        condition.release()


def do_print():
    if condition.acquire():
        while product_list is None:
            condition.wait()
        for i in product_list:
            common_logger.info(i)
        condition.release()


def do_create():
    global product_list
    if condition.acquire():
        if product_list is None:
            product_list = [0 for i in range(10)]
            condition.notifyAll()
        condition.release()


def test_producer_consumer2():
    """
    测试生产者消费者。
    :return:
    """
    t_set = threading.Thread(target=do_set, name='tset')
    t_print = threading.Thread(target=do_print, name='tprint')
    t_create = threading.Thread(target=do_create, name='tcreate')
    t_set.start()
    t_print.start()
    t_create.start()



if __name__ == "__main__":
    # loop_action()
    # test_multi_cpu_threading()
    # test_thread_start()
    # test_thread_join()
    # test_thread_local()
    # test_balance()
    test_balance_with_lock()
    pass
