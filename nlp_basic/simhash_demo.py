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


class NewsSimHash(object):
    """
    新闻信息的simhash示例。
    """
    def __init__(self, origin_news_map_file, result_file_path, ):
        """
        构造函数。
        """
        self.bucket = collections.defaultdict(set)
        self.origin_news_map_file = origin_news_map_file
        self.test_news_fp = {}
        self.lib_news_fp = {}
        pass

    def process(self):
        """
        执行过程
        :return:
        """
        pass

    def load_origin_news_map(self):
        """
        加载原始新闻映射文件。文件中包含url链接和文件路径。
        :return:
        """
        fin = codecs.open(self.origin_news_map_file, 'r', 'utf-8')
        for line in fin:
            lines = line.strip()
            if len(lines) == 0:
                continue

            arr = lines.split('\t')
            if len(arr) < 3:
                continue
            self.lib_news_fp[arr[0]] = arr[3]



# 读入库中存储的所有新闻
lib_newsfp_file = sys.argv[1]
result_file = sys.argv[2]

test_news_fp = {}
lib_news_fp = {}

bucket = collections.defaultdict(set)

offsets = []


def cacu_frequent(list1):
    frequent = {}
    for i in list1:
        if i not in frequent:
            frequent[i] = 0
        frequent[i] += 1
    return frequent


def load_lib_newsfp_file():
    global lib_news_fp

    fin = codecs.open(lib_newsfp_file, 'r', 'utf-8')
    for line in fin:
        lines = line.strip()
        if len(lines) == 0:
            continue
        Arr = lines.split('\t')

        if len(Arr) < 3:
            continue
        lib_news_fp[Arr[0]] = Arr[3]


def get_near_dups(check_value):
    ans = set()

    for key in get_keys(int(check_value)):
        dups = bucket[key]
        for dup in dups:
            total_value, url = dup.split(',', 1)
            if isSimilar(int(check_value), int(total_value)) == True:
                ans.add(url)
                break  # 与一条重复 退出查找
        if ans:
            break

    return list(ans)


def ini_Index():
    global bucket

    getoffsets()
    print
    offsets
    objs = [(str(url), str(values)) for url, values in lib_news_fp.items()]

    for i, q in enumerate(objs):
        addindex(*q)


def addindex(url, value):
    global bucket
    for key in get_keys(int(value)):
        v = '%d,%s' % (int(value), url)
        bucket[key].add(v)


def deleteindex(url, value):
    global bucket
    for key in get_keys(int(value)):
        v = '%d,%s' % (int(value), url)
        if v in bucket[key]:
            bucket[key].remove(v)


def getoffsets(f=64, k=4):
    global offsets

    offsets = [f // (k + 1) * i for i in range(k + 1)]


def get_keys(value, f=64):
    for i, offset in enumerate(offsets):
        if i == (len(offsets) - 1):
            m = 2 ** (f - offset) - 1
        else:
            m = 2 ** (offsets[i + 1] - offset) - 1
        c = value >> offset & m
        yield '%x:%x' % (c, i)


def bucket_size():
    return len(bucket)


def isSimilar(value1, value2, n=4, f=64):
    ans = 0
    x = (value1 ^ value2) & ((1 << f) - 1)
    while x and (ans <= n):
        ans += 1
        x &= x - 1
    if ans <= n:
        return True
    return False


def load_test_file():
    global test_news_fp

    for line in sys.stdin:
        features = []

        result = line.strip().split('\t')

        url = result[0]
        content = result[2].split()
        title = result[1].split()
        features.extend(content)
        features.extend(title)
        total_features = cacu_frequent(features)

        test_news_fp[url] = build_by_features(total_features)


def load_test_newsfp_file():
    global test_news_fp

    for line in sys.stdin:
        lines = line.strip()
        if len(lines) == 0:
            continue
        Arr = lines.split('\t')

        if len(Arr) < 3:
            continue
        test_news_fp[Arr[0]] = Arr[3]


def build_by_features(features, f=64, hashfunc=None):
    v = [0] * f
    masks = [1 << i for i in range(f + f)]
    if hashfunc is None:
        def _hashfunc(x):
            return int(hashlib.md5(x).hexdigest(), 16)

        hashfunc = _hashfunc
    if isinstance(features, dict):
        total_features = features.items()
    else:
        total_features = features

    for fea in total_features:
        if isinstance(fea, str):
            h = hashfunc(fea.encode('utf-8'))
            w = 1
        else:
            h = hashfunc(fea[0].encode('utf-8'))
            w = fea[1]
        for i in range(f):
            v[i] += w if h & masks[i + 32] else -w
    ans = 0

    for i in range(f):
        if v[i] >= 0:
            ans |= masks[i]
    return ans


sum = 0


def process():
    global test_news_fp
    global sum


fout = codecs.open(result_file, 'w', 'utf-8')

load_lib_newsfp_file()
#   load_test_file()
ini_Index()
check_features = test_news_fp.items()
lib_features = lib_news_fp.items()
i = 0
for check_fp in check_features:
    #       print i
    ans = []
    ans = get_near_dups(check_fp[1])
    if ans:
        for url in ans:
            output_str = str(check_fp[0]) + '\t' + str(url)
            fout.write(output_str + '\n')
            # break
            # print check_fp[0],'is duplicate'
        sum = sum + 1  # del test_news_fp[check_fp[0]]
        print
        i

    i += 1
fout.close()


def test_news_simhash():
    """
    测试新闻信息内容的simhash。
    :return:
    """
    #        process()
    begin = datetime.datetime.now()
    load_test_newsfp_file()
    #   load_test_file()
    #   getoffsets()
    #   print offsets
    #   load_lib_newsfp_file()
    process()

    end = datetime.datetime.now()
    common_logger.info("耗时：{0} 重复新闻数：{1} 准确率：{2}"
                       .format(end - begin, sum, sum / 2589))


if __name__ == '__main__':
    test_simhash()
