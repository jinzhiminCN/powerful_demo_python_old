# -*- coding:utf-8 -*-

# ==============================================================================
# 协同过滤推荐算法的示例。
# 协同过滤CF(Collaborative Filtering).
# 基于用户的协同过滤算法UBCF（User-Based Collaborative Filtering）
# 基于商品(Item)的协同过滤算法IBCF（Item-Based Collaborative Filtering）
# ==============================================================================
import random
import math
import time
import os
import config.common_config as com_config
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()
# 数据文件名
data_file_name = "/ml-100k/u.data"
# 数据文件路径
data_file_path = os.path.join(com_config.RESOURCE_DIR, data_file_name)


class UserBasedCF:
    def __init__(self, datafile=None):
        self.datafile = datafile
        self.data = None
        self.train_data = None
        self.test_data = None
        self.user_sim = None
        self.user_sim_best = None

        self.read_data()
        self.split_data(3, 47)

    def read_data(self, datafile=None):
        """
        read the data from the data file which is a data set.
        把文件中的内容读到data中。
        """
        self.datafile = datafile or self.datafile
        self.data = []
        for line in open(self.datafile):
            user_id, item_id, record, _ = line.split()
            self.data.append((user_id, item_id, int(record)))

    def split_data(self, k, seed, data=None, m=8):
        """
        split the data set
        test_data is a test data set
        train_data is a train set
        test data set / train data set is 1:m-1
        拆分数据集为训练集和测试集。
        :param k: 随机数为k时，拆分到测试集中。
        :param seed: 随机种子
        :param data: 数据集
        :param m: 拆分比例，训练集和测试集的比例为m。
        """
        self.test_data = {}
        self.train_data = {}
        self.data = data or self.data
        random.seed(seed)
        for user, item, record in self.data:
            if random.randint(0, m) == k:
                self.test_data.setdefault(user, {})
                self.test_data[user][item] = record
            else:
                self.train_data.setdefault(user, {})
                self.train_data[user][item] = record

    def user_similarity(self, train=None):
        """
        计算用户相似性。
        :param train:
        :return:
        """
        train = train or self.train_data
        self.user_sim = dict()
        for u in train.keys():
            for v in train.keys():
                if u == v:
                    continue
                self.user_sim.setdefault(u, {})
                self.user_sim[u][v] = len(set(train[u].keys()) & set(train[v].keys()))
                self.user_sim[u][v] /= math.sqrt(len(train[u]) * len(train[v]) * 1.0)

    def user_similarity_best(self, train=None):
        """
        the other method of getting user similarity which is better than above
        you can get the method on page 46
        In this experiment，we use this method
        """
        train = train or self.train_data
        self.user_sim_best = dict()
        item_users = dict()
        for u, item in train.items():
            for i in item.keys():
                item_users.setdefault(i, set())
                item_users[i].add(u)
        user_item_count = dict()
        count = dict()
        for item, users in item_users.items():
            for u in users:
                user_item_count.setdefault(u, 0)
                user_item_count[u] += 1
                for v in users:
                    if u == v: continue
                    count.setdefault(u, {})
                    count[u].setdefault(v, 0)
                    count[u][v] += 1
        for u, related_users in count.items():
            self.user_sim_best.setdefault(u, dict())
            for v, cuv in related_users.items():
                self.user_sim_best[u][v] = cuv / math.sqrt(user_item_count[u] * user_item_count[v] * 1.0)

    def recommend(self, user, train=None, k=8, nitem=40):
        train = train or self.train_data
        rank = dict()
        interacted_items = train.get(user, {})
        for v, wuv in sorted(self.user_sim_best[user].items(), key=lambda x: x[1], reverse=True)[0:k]:
            for i, rvi in train[v].items():
                if i in interacted_items:
                    continue
                rank.setdefault(i, 0)
                rank[i] += wuv * rvi
        return dict(sorted(rank.items(), key=lambda x: x[1], reverse=True)[0:nitem])

    def recall_and_precision(self, train=None, test=None, k=8, nitem=10):
        """
        Get the recall and precision, the method you want to know is listed
        in the page 43
        """
        train = train or self.train_data
        test = test or self.test_data
        hit = 0
        recall = 0
        precision = 0
        for user in train.keys():
            tu = test.get(user, {})
            rank = self.recommend(user, train=train, k=k, nitem=nitem)
            for item, _ in rank.items():
                if item in tu:
                    hit += 1
            recall += len(tu)
            precision += nitem
        return hit / (recall * 1.0), hit / (precision * 1.0)

    def coverage(self, train=None, test=None, k=8, nitem=10):
        train = train or self.train_data
        test = test or self.test_data
        recommend_items = set()
        all_items = set()
        for user in train.keys():
            for item in train[user].keys():
                all_items.add(item)
            rank = self.recommend(user, train, k=k, nitem=nitem)
            for item, _ in rank.items():
                recommend_items.add(item)
        return len(recommend_items) / (len(all_items) * 1.0)

    def popularity(self, train=None, test=None, k=8, nitem=10):
        """
        Get the popularity
        the algorithm on page 44
        """
        train = train or self.train_data
        test = test or self.test_data
        item_popularity = dict()
        for user, items in train.items():
            for item in items.keys():
                item_popularity.setdefault(item, 0)
                item_popularity[item] += 1
        ret = 0
        n = 0
        for user in train.keys():
            rank = self.recommend(user, train, k=k, nitem=nitem)
            for item, _ in rank.items():
                ret += math.log(1 + item_popularity[item])
                n += 1
        return ret / (n * 1.0)


class ItemBasedCF(object):
    def __init__(self, datafile=None):
        self.datafile = datafile
        self.data = None
        self.train_data = None
        self.test_data = None
        self.item_sim = None
        self.item_sim_best = None

        self.read_data(self.datafile)
        self.split_data(3, 47)

    def read_data(self, datafile=None):
        """
        read the data from the data file which is a data set
        """
        print("ireadData:")
        self.datafile = datafile or self.datafile
        self.data = []
        for line in open(self.datafile):
            user_id, item_id, record, _ = line.split()
            self.data.append((user_id, item_id, int(record)))
            #      格式 [('196', '242', 3), ('186', '302', 3), ('22', '377', 1)]

    def split_data(self, k, seed, data=None, m=8):
        """
        split the data set
        test_data is a test data set
        train_data is a train set
        test data set / train data set is 1:M-1
        """
        self.test_data = {}
        self.train_data = {}
        data = data or self.data
        random.seed(seed)
        for user, item, record in self.data:
            if random.randint(0, m) == k:
                self.test_data.setdefault(user, {})
                self.test_data[user][item] = record
            else:
                self.train_data.setdefault(user, {})
                self.train_data[user][item] = record
                # print(self.testdata)
                #        格式{'291': {'1042': 4, '118': 2}, '200': {'222': 5},
                # '308': {'1': 4}, '167': {'486': 4}, '122': {'387': 5}, '210': {'40': 3},

    def item_similarity(self, train=None):
        """
        the other method of getting user similarity which is better than above
        you can get the method on page 46
        In this experiment，we use this method
        """
        train = train or self.train_data
        self.item_sim_best = dict()
        N = dict()
        C = dict()
        for u, items in train.items():
            for i in items:
                N.setdefault(i, 0)
                N[i] += 1
                for j in items:
                    if i == j:
                        continue
                    C.setdefault(i, {})
                    C[i].setdefault(j, 0)
                    C[i][j] += 1
        for i, related_items in C.items():
            self.item_sim_best.setdefault(i, dict())
            for j, cij in related_items.items():
                self.item_sim_best[i][j] = cij / math.sqrt(N[i] * N[j] * 1.0)

    def recommend(self, user, train=None, k=8, nitem=10):
        train = train or self.train_data
        rank = dict()
        ru = train.get(user, {})
        for i, pi in ru.items():
            for j, wj in sorted(self.item_sim_best[i].items(), key=lambda x: x[1], reverse=True)[0:k]:
                if j in ru:
                    continue
                rank.setdefault(j, 0)
                rank[j] += pi * wj
        return dict(sorted(rank.items(), key=lambda x: x[1], reverse=True)[0:nitem])

    def recall_and_precision(self, train=None, test=None, k=8, nitem=10):
        """
        Get the recall and precision, the method you want to know is listed
        in the page 43
        """
        train = train or self.train_data
        test = test or self.test_data
        hit = 0
        recall = 0
        precision = 0
        for user in train.keys():
            tu = test.get(user, {})
            rank = self.recommend(user, train=train, k=k, nitem=nitem)
            for item, _ in rank.items():
                if item in tu:
                    hit += 1
            recall += len(tu)
            precision += nitem
        return hit / (recall * 1.0), hit / (precision * 1.0)

    def coverage(self, train=None, test=None, k=8, nitem=10):
        train = train or self.train_data
        test = test or self.test_data
        recommend_items = set()
        all_items = set()
        for user in train.keys():
            for item in train[user].keys():
                all_items.add(item)
            rank = self.recommend(user, train, k=k, nitem=nitem)
            for item, _ in rank.items():
                recommend_items.add(item)
        return len(recommend_items) / (len(all_items) * 1.0)

    def popularity(self, train=None, test=None, k=8, nitem=10):
        """
        Get the popularity
        the algorithm on page 44
        """
        train = train or self.train_data
        test = test or self.test_data
        item_popularity = dict()
        for user, items in train.items():
            for item in items.keys():
                item_popularity.setdefault(item, 0)
                item_popularity[item] += 1
        ret = 0
        n = 0
        # 对每一个user进行推荐 计算其流行度
        for user in train.keys():
            rank = self.recommend(user, train, k=k, nitem=nitem)
            for item, _ in rank.items():
                ret += math.log(1 + item_popularity[item])
                n += 1
        return ret / (n * 1.0)


def test_ubcf_recommend():
    ubcf = ItemBasedCF(data_file_path)
    ubcf.read_data()
    ubcf.split_data(4, 100)
    ubcf.item_similarity()
    user = "345"
    rank = ubcf.recommend(user, k=3)
    for i, rvi in rank.items():
        items = ubcf.test_data.get(user, {})
        record = items.get(i, 0)
        print("%5s: %.4f--%.4f" % (i, rvi, record))


def test_user_based_cf():
    start_time = time.clock()
    cf = UserBasedCF(data_file_path)
    cf.user_similarity_best()
    print("%3s%20s%20s%20s%20s%20s" % ('K', "recall", 'precision', 'coverage', 'popularity', 'time'))
    for k in [5, 10, 20, 40, 80, 160]:
        recall, precision = cf.recall_and_precision(k=k)
        coverage = cf.coverage(k=k)
        popularity = cf.popularity(k=k)
        print("%3d%19.3f%%%19.3f%%%19.3f%%%20.3f%19.3fs" % (
        k, recall * 100, precision * 100, coverage * 100, popularity, time.clock() - start_time))


def test_ibcf_recommend():
    ibcf = ItemBasedCF(data_file_path)
    ibcf.read_data()
    ibcf.split_data(4, 100)
    ibcf.item_similarity()
    user = "345"
    rank = ibcf.recommend(user, k=3)
    for i, rvi in rank.items():
        items = ibcf.test_data.get(user, {})
        record = items.get(i, 0)
        print("%5s: %.4f--%.4f" % (i, rvi, record))


def test_item_based_cf():
    start_time = time.clock()
    cf = ItemBasedCF(data_file_path)
    cf.item_similarity()
    print("%3s%20s%20s%20s%20s%20s" % ('K', "recall", 'precision', 'coverage', 'popularity', 'time'))
    for k in [5, 10, 20, 40, 80, 160]:
        recall, precision = cf.recall_and_precision(k=k)
        coverage = cf.coverage(k=k)
        popularity = cf.popularity(k=k)
        print("%3d%19.3f%%%19.3f%%%19.3f%%%20.3f%19.3fs" % (
        k, recall * 100, precision * 100, coverage * 100, popularity, time.clock() - start_time))


if __name__ == "__main__":
    test_item_based_cf()
    test_user_based_cf()



