# -*- coding:utf-8 -*-

# ==============================================================================
# 最简单的推荐系统的示例。
# ==============================================================================
from math import sqrt
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()

# 简单评分数据
critics = {
    'Cathy': {'item_1': 2.5, 'item_2': 3.5, 'item_3': 3, 'item_4': 3.5, 'item_5': 2.5, 'item_5': 3},
    'Sophie': {'item_1': 3, 'item_2': 3.5, 'item_3': 1.5, 'item_4': 5, 'item_5': 1.5, 'item_5': 3},
    'Susie': {'item_1': 2.5, 'item_2': 3, 'item_4': 3.5, 'item_5': 4},
    'Antonio': {'item_2': 3.5, 'item_3': 3, 'item_4': 4, 'item_5': 2.5, 'item_5': 4.5},
    'Marco': {'item_1': 3, 'item_2': 4, 'item_3': 2, 'item_4': 3, 'item_5': 2, 'item_5': 3},
    'Jack': {'item_1': 3, 'item_2': 4, 'item_4': 5, 'item_5': 3.5, 'item_5': 3},
    'Leo': {'item_2': 4.5, 'item_4': 4, 'item_5': 1.0}
}


def sim_distance(prefers, person1, person2):
    """
    计算有关person1与person2欧几里得距离的相似度。
    :param prefers: 用户偏好数据
    :param person1: 用户1
    :param person2: 用户2
    :return:
    """
    # 得到shared_items的列表
    si = {}
    for item in prefers[person1]:
        if item in prefers[person2]:
            si[item] = 1

    # 如果两者没有共同之处，则返回0
    if len(si) == 0:
        return 0

    # 计算所有差值的平方和
    sum_of_squares = sum([pow(prefers[person1][item] - prefers[person2][item], 2)
                          for item in prefers[person1] if item in prefers[person2]])
    return 1 / (1 + sqrt(sum_of_squares))


def sim_pearson(prefers, person1, person2):
    """
    计算p1和p2的皮尔逊相关系数。
    :param prefers: 用户偏好数据
    :param person1: 用户1
    :param person2: 用户2
    :return:
    """
    # 得到shared_items的列表
    si = {}
    for item in prefers[person1]:
        if item in prefers[person2]:
            si[item] = 1

    n = len(si)
    if n == 0:
        return 1

    sum1 = sum([prefers[person1][it] for it in si])
    sum2 = sum([prefers[person2][it] for it in si])

    sum1_sq = sum([pow(prefers[person1][it], 2) for it in si])
    sum2_sq = sum([pow(prefers[person2][it], 2) for it in si])

    p_sum = sum([prefers[person1][it] * prefers[person2][it] for it in si])
    # calculate the pearson value
    num = p_sum - (sum1 * sum2 / n)
    den = sqrt((sum1_sq - pow(sum1, 2) / n) * (sum2_sq - pow(sum2, 2) / n))
    if den == 0:
        return 0

    result = num / den

    return result


def top_matches(prefers, person, n, similarity=sim_pearson):
    """
    计算相似度最高的top N。
    :param prefers:
    :param person:
    :param n:
    :param similarity:
    :return:
    """
    scores = [(similarity(prefers, person, other), other)
              for other in prefers if other != person]
    # sort the similarity
    scores.sort()
    scores.reverse()
    return scores[0:n]


def test_sim_person_distance():
    """
    测试用户距离。
    :return:
    """
    sim_dist = sim_distance(critics, "Cathy", "Sophie")
    common_logger.info("distance:{0}".format(sim_dist))
    sim_prson = sim_pearson(critics, "Cathy", "Sophie")
    common_logger.info("pearson:{0}".format(sim_prson))
    top_leo1 = top_matches(critics, "Leo", 5, similarity=sim_distance)
    common_logger.info("pearson:{0}".format(top_leo1))
    top_leo2 = top_matches(critics, "Leo", 5, similarity=sim_pearson)
    common_logger.info("pearson:{0}".format(top_leo2))


if __name__ == '__main__':
    test_sim_person_distance()
    pass
