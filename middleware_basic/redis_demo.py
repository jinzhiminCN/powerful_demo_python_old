# -*- coding:utf-8 -*-

# ==============================================================================
# 测试redis的基本用法。
# 1. 连接方式
#     redis-py提供两个类Redis和StrictRedis用于实现Redis的命令，StrictRedis用于实现大部分官方的命令，
# 并使用官方的语法和命令，Redis是StrictRedis的子类。
# ==============================================================================
import redis
from rediscluster import StrictRedisCluster
from rediscluster.exceptions import RedisClusterException
from config.common_config import *
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()
# redis主机和端口
host = "127.0.0.1"
port = 6379
redis_hosts = ["127.0.0.1:6379", "127.0.0.1:6378", "127.0.0.1:6377"]
redis_nodes = [dict(zip(['host', 'port'], host_item.split(':'))) for host_item in redis_hosts]
redis_pass = "999999"


def test_redis_connect():
    """
    测试redis链接。
    :return:
    """
    rds = redis.Redis(host=host, port=port, db=0)
    rds.set('name', 'nike')


def test_redis_cluster_connect():
    """
    测试redis集群链接。
    :return:
    """
    try:
        rds_cluster = StrictRedisCluster(startup_nodes=redis_nodes,
                                         password=redis_pass,
                                         decode_responses=True)
    except RedisClusterException:
        rds_cluster = StrictRedisCluster(startup_nodes=redis_nodes,
                                         decode_responses=True)
    value = rds_cluster.hget('my_hash', "my_key")
    common_logger.info(value)


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


def test_redis_string():
    """
    测试redis的字符串操作。
    :return:
    """
    rds = get_redis_connect()
    # 在Redis中设置值，默认不存在则创建，存在则修改
    rds.set('name', 'forever')

    # 设置过期时间（秒）
    rds.setex('name_second', "60 seconds", 60)

    # 设置过期时间（豪秒）
    rds.psetex("name_millisecond", 10000, "10000 mSec")

    # 批量设置值
    rds.mset(name1='name1', name2='name2')
    rds.mset({"name3": "name3", "name4": "name4"})

    # 批量获取
    common_logger.info(rds.mget("name1", "name2"))
    name_list = ["name3", "name2"]
    common_logger.info(rds.mget(name_list))

    # 返回name对应值的字节长度（一个汉字3个字节）
    rds.set("name", "google")
    common_logger.info("google len:{0}".format(rds.strlen("name")))

    # 对二进制表示位进行操作
    int_str = "345"
    rds.set("name", int_str)
    for i in int_str:
        # 输出 值、ASCII码中对应的值、对应值转换的二进制
        common_logger.info("{0}, {1}, {2}".format(i, ord(i), bin(ord(i))))
    # 把第7位改为0，也就是3对应的变成了0b110001
    rds.setbit("name", 6, 0)
    common_logger.info(rds.get("name"))

    # 自增mount对应的值，当mount不存在时，则创建mount＝amount，否则，则自增,amount为自增数(整数)
    common_logger.info(rds.incr("mount", amount=2))
    common_logger.info(rds.incr("mount"))
    common_logger.info(rds.incr("mount", amount=3))
    common_logger.info(rds.incr("mount", amount=6))
    common_logger.info(rds.get("mount"))


def test_redis_hash():
    """
    测试redis的hash操作。
    :return:
    """
    rds = get_redis_connect()
    # name对应的hash中设置一个键值对（不存在，则创建，否则，修改）
    rds.hset("dic_name", "a1", "aa")
    # 在name对应的hash中根据key获取value
    common_logger.info(rds.hget("dic_name","a1"))
    # 在name对应的hash中批量设置键值对,mapping:字典
    dic = {"a1": "aa", "b1": "bb"}
    rds.hmset("dic_name", dic)
    # 获取name对应hash的所有键值
    common_logger.info(rds.hgetall("dic_name"))
    # 获取hash中键值对的个数
    common_logger.info(rds.hlen("dic_name"))
    # 获取hash中所有的key的值
    common_logger.info(rds.hkeys("dic_name"))
    # 获取hash中所有的value的值
    common_logger.info(rds.hvals("dic_name"))
    # 检查name对应的hash是否存在当前传入的key
    common_logger.info(rds.hexists("dic_name", "a1"))
    # 删除指定name对应的key所在的键值对
    common_logger.info(rds.hdel("dic_name", "a1"))


def test_redis_list():
    """
    测试redis的list操作。
    :return:
    """
    rds = get_redis_connect()
    # 保存在列表中的顺序为5，4，3，2
    rds.lpush("list_name", 2)
    rds.lpush("list_name", 3, 4, 5)

    # 在name对应的list中添加元素，只有name已经存在时，值添加到列表的最左边
    rds.lpushx("list_push", "push_left_1")

    # 对list中的某一个索引位置重新赋值
    rds.lset("list_name", 0, "bbb")

    # 删除name对应的list中的指定值
    rds.lrem("list_name", "SS", num=0)

    # name对应的list元素的个数
    common_logger.info(rds.llen("list_name"))

    # 在name对应的列表的某一个值前或后插入一个新值
    # 在列表内找到第一个元素2，在它前面插入SS
    rds.linsert("list_name", "BEFORE", "2", "SS")

    # 移除列表的左侧第一个元素，返回值则是第一个元素
    common_logger.info(rds.lpop("list_name"))
    # 根据索引获取列表内元素
    common_logger.info(rds.lindex("list_name", 1))
    # 分片获取元素
    common_logger.info(rds.lrange("list_name", 0, -1))

    # 移除列表内没有在该索引之内的值
    rds.ltrim("list_name", 0, 2)

    # 从一个列表取出最右边的元素，同时将其添加至另一个列表的最左边
    rds.rpoplpush("list_name", "list_name1")

    # 同rpoplpush，多了个timeout, timeout：取数据的列表没元素后的阻塞时间，0为一直阻塞
    rds.brpoplpush("list_name", "list_name1", timeout=0)


def test_redis_set():
    """
    测试redis的set操作。
    :return:
    """
    rds = get_redis_connect()
    # 给name对应的集合中添加元素
    rds.sadd("set_name", "aa")
    rds.sadd("set_name", "cc", "bb")
    # 获取name对应的集合的所有成员
    common_logger.info(rds.smembers("set_name"))
    # 获取name对应的集合中的元素个数
    common_logger.info(rds.scard("set_name"))

    # 在第一个name对应的集合中且不在其他name对应的集合的元素集合
    rds.sadd("set_name", "aa", "bb")
    rds.sadd("set_name1", "bb", "cc")
    rds.sadd("set_name2", "bb", "cc", "dd")
    common_logger.info(rds.sdiff("set_name", "set_name1", "set_name2"))
    # 获取多个name对应集合的交集
    common_logger.info(rds.sinter("set_name", "set_name1", "set_name2"))
    # 获取多个name对应的集合的并集
    common_logger.info(rds.sunion("set_name", "set_name1", "set_name2"))
    # 删除name对应的集合中的某些值
    common_logger.info(rds.srem("set_name2", "bb", "dd"))
    # 从集合的随机移除一个元素，并将其返回
    common_logger.info(rds.spop("set_name"))


def test_redis_zset():
    """
    测试redis的有序集合：
    在集合的基础上，为每元素排序，元素的排序需要根据另外一个值来进行比较。
    所以，对于有序集合，每一个元素有两个值，即：值和分数，分数专门用来做排序。
    :return:
    """
    rds = get_redis_connect()
    # 在name对应的有序集合中添加元素
    rds.zadd("zset_name", "a1", 6, "a2", 2, "a3", 5)
    # 或
    rds.zadd('zset_name1', b1=10, b2=5)

    # 获取有序集合中分数在[min,max]之间的个数
    common_logger.info(rds.zcount("zset_name", 1, 5))
    # 获取有序集合内元素的数量
    common_logger.info("size:{0}".format(rds.zcard("zset_name")))

    # 自增有序集合内value对应的分数
    rds.zincrby("zset_name", "a1", amount=2)
    # 按照索引范围获取name对应的有序集合的元素
    aa = rds.zrange("zset_name", 0, 1, desc=False, withscores=True, score_cast_func=int)
    common_logger.info(aa)

    # 获取value值在name对应的有序集合中的排行位置（从0开始）
    common_logger.info(rds.zrank("zset_name", "a2"))
    common_logger.info(rds.zrevrank("zset_name", "a2"))

    # 获取name对应有序集合中 value 对应的分数
    common_logger.info(rds.zscore("zset_name", "a1"))


if __name__ == "__main__":
    # test_redis_connect()
    # test_redis_pool()
    # test_redis_pipe()
    # test_redis_publish()
    # test_redis_subscribe()
    # test_redis_publish()
    # test_redis_list()
    # test_redis_string()
    # test_redis_hash()
    # test_redis_set()
    # test_redis_zset()
    test_redis_cluster_connect()
    pass

