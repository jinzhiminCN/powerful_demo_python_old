# _*_coding:utf-8_*_

# ==============================================================================
# 测试simhash算法。
# Google Moses Charikar发表的论文“detecting near-duplicates for web crawling”
# 中提出simhash算法，专门用来解决亿万级别的网页的去重任务。
# simhash作为locality sensitive hash（局部敏感哈希）的一种：
# 其主要思想是降维，将高维的特征向量映射成低维的特征向量，通过两个向量的Hamming Distance来确定文章是否重复或者高度近似。
# simhash算法分为5个步骤：分词、hash、加权、合并、降维，具体过程如下所述：
# 1. 分词
# 给定一段语句，进行分词，得到有效的特征向量，然后为每一个特征向量设置1-5等5个级别的权重
# （如果是给定一个文本，那么特征向量可以是文本中的词，其权重可以是这个词出现的次数）。
# 2. hash
# 通过hash函数计算各个特征向量的hash值，hash值为二进制数01组成的n-bit签名。
# 3. 加权
# 在hash值的基础上，给所有特征向量进行加权，即W = Hash * weight，且遇到1则hash值和权值正相乘，
# 遇到0则hash值和权值负相乘。
# 4. 合并
# 将上述各个特征向量的加权结果累加，变成只有一个序列串。
# 5. 降维
# 对于n-bit签名的累加结果，如果大于0则置1，否则置0，从而得到该语句的simhash值。
#
# 海明距离的求法：异或时，只有在两个比较的位不同时其结果是1 ，否则结果为0，
# 两个二进制“异或”后得到1的个数即为海明距离的大小。
# 每篇文档得到SimHash签名值后，接着计算两个签名的海明距离即可。
# 根据经验值，对64位的 SimHash值，海明距离在3以内的可认为相似度比较高。
# ==============================================================================
import sys
import re
import hashlib
import collections
import datetime
import codecs
import itertools
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()


class SimHash:
    def __init__(self, tokens='', hash_bits=128):
        """
        构造函数。
        :param tokens:
        :param hash_bits:
        """
        self.hash_bits = hash_bits
        self.hash = self.simhash(tokens)

    def __str__(self):
        """
        toString函数。
        :return:
        """
        return str(self.hash)

    def simhash(self, tokens):
        """
        生成simhash值。
        :param tokens:
        :return:
        """
        v = [0] * self.hash_bits

        # t为token的普通hash值
        for t in [self._string_hash(x) for x in tokens]:
            for i in range(self.hash_bits):
                bit_mask = 1 << i
                if t & bit_mask:
                    v[i] += 1  # 查看当前bit位是否为1,是的话将该位+1
                else:
                    v[i] -= 1  # 否则的话,该位-1
        fingerprint = 0
        for i in range(self.hash_bits):
            if v[i] >= 0:
                fingerprint += 1 << i

        # 整个文档的fingerprint为最终各个位>=0的和
        return fingerprint

    def hamming_distance(self, other):
        """
        求海明距离。
        :param other:
        :return:
        """
        x = (self.hash ^ other.hash) & ((1 << self.hash_bits) - 1)
        tot = 0
        while x:
            tot += 1
            x &= x - 1
        return tot

    def similarity(self, other):
        """
        求相似度。
        :param other:
        :return:
        """
        a = float(self.hash)
        b = float(other.hash)
        if a > b:
            return b / a
        else:
            return a / b

    def _string_hash(self, source):
        """
        针对source生成hash值 (一个可变长度版本的Python的内置散列)
        :param source:
        :return:
        """
        if source == "":
            return 0
        else:
            x = ord(source[0]) << 7
            m = 1000003
            mask = 2 ** self.hash_bits - 1
            for c in source:
                x = ((x * m) ^ ord(c)) & mask
            x ^= len(source)
            if x == -1:
                x = -2
            return x


def test_simhash():
    """
    测试simhash。
    :return:
    """
    s = 'This is a test string for testing'
    hash1 = SimHash(s.split())

    s = 'This is a test string for testing also'
    hash2 = SimHash(s.split())

    s = 'nai nai ge xiong cao'
    hash3 = SimHash(s.split())

    common_logger.info("{0}   {1}".format(hash1.hamming_distance(hash2), hash1.similarity(hash2)))
    common_logger.info("{0}   {1}".format(hash1.hamming_distance(hash3), hash1.similarity(hash3)))


def is_similar(value1, value2, n=4, f=64):
    """
    比较相似性。
    :param value1:
    :param value2:
    :param n:
    :param f:
    :return:
    """
    ans = 0
    x = (value1 ^ value2) & ((1 << f) - 1)
    while x and (ans <= n):
        ans += 1
        x &= x - 1
    if ans <= n:
        return True
    return False


def test_similar():
    """
    测试相似性。
    :return:
    """
    value1 = 55
    value2 = 56
    result = is_similar(value1, value2)
    common_logger.info("{0}".format(result))


if __name__ == '__main__':
    # test_simhash()
    pass
    test_similar()