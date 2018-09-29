# -*- coding:utf-8 -*-

# ==============================================================================
# 测试redis的基本用法。
# 1. 连接方式
#     redis-py提供两个类Redis和StrictRedis用于实现Redis的命令，StrictRedis用于实现大部分官方的命令，
# 并使用官方的语法和命令，Redis是StrictRedis的子类。
# ==============================================================================
import redis
from config.common_config import *
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()
host = "127.0.0.1"
port = 6379


def test_redis_connect():
    """
    测试redis链接。
    :return:
    """
    rds = redis.Redis(host=host, port=port, db=0)
    rds.set('name', 'nike')


def test_redis_pool():
    """
    测试redis连接池。
    redis-py使用connection pool来管理对一个redis server的所有连接，避免每次建立、释放连接的开销。
    默认，每个Redis实例都会维护一个自己的连接池。可以直接建立一个连接池，然后作为参数Redis，
    这样就可以实现多个Redis实例共享一个连接池。
    :return:
    """
    pool = redis.ConnectionPool(host='localhost', port=port, decode_responses=True)

    rds1 = redis.Redis(connection_pool=pool)
    rds2 = redis.Redis(connection_pool=pool)
    rds1.set('apple', 'a')
    common_logger.info(rds1.get('apple'))
    rds2.set('banana', 'b')
    common_logger.info(rds2.get('banana'))

    # 打印当前连接的情况
    common_logger.info(rds1.client_list())
    common_logger.info(rds2.client_list())


def get_redis_connect():
    """
    获取redis连接。
    :return:
    """
    pool = redis.ConnectionPool(host='localhost', port=port, decode_responses=True)
    rds = redis.Redis(connection_pool=pool)
    return rds


def test_redis_pipe():
    """
    测试redis管道。
    redis-py默认在执行每次请求都会创建（连接池申请连接）和断开（归还连接池）一次连接操作，
    如果想要在一次请求中指定多个命令，则可以使用pipeline实现一次请求指定多个命令，
    并且默认情况下一次pipeline是原子性操作。transaction参数指示是否所有的命令都以原子方式执行。
    :return:
    """
    rds = get_redis_connect()

    pipe = rds.pipeline(transaction=True)
    pipe.set('name', 'google')
    pipe.set('name', 'nike')
    # 执行事务
    pipe.execute()


class RedisHelper(object):
    """
    Redis发布和订阅，定义频道为monitor，定义发布(publish)及订阅(subscribe)方法。
    """
    def __init__(self):
        self.__conn = get_redis_connect()
        # 定义名称
        self.channel = 'monitor'

    def publish(self, msg):
        """
        定义发布方法。
        :param msg:
        :return:
        """
        self.__conn.publish(self.channel, msg)
        return True

    def subscribe(self):
        """
        定义订阅方法。
        """
        pub = self.__conn.pubsub()
        pub.subscribe(self.channel)
        pub.parse_response()
        return pub


def test_redis_publish():
    """
    测试redis发布信息。
    :return:
    """
    obj = RedisHelper()
    obj.publish('hello')


def test_redis_subscribe():
    """
    测试redis订阅信息。
    :return:
    """
    obj = RedisHelper()
    redis_sub = obj.subscribe()

    while True:
        msg = redis_sub.parse_response()
        # 此处的信息格式['消息类型', '频道', '消息']
        common_logger.info(msg)


def test_redis_list():
    """
    测试redis的list操作。
    :return:
    """
    rds = get_redis_connect()
    # 保存在列表中的顺序为5，4，3，2
    rds.lpush("list_name", 2)
    rds.lpush("list_name", 3, 4, 5)


if __name__ == "__main__":
    # test_redis_connect()
    # test_redis_pool()
    # test_redis_pipe()
    # test_redis_publish()
    # test_redis_subscribe()
    # test_redis_publish()
    test_redis_list()
    pass

