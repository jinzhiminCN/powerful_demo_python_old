# _*_coding:utf-8_*_

# ==============================================================================
# 测试jieba分词的相关方法。
# ==============================================================================
import jieba
import jieba.analyse
import jieba.posseg as posseg
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()


def test_segment():
    """
    测试简单分词方法。
    :return:
    """
    # 全模式
    seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
    common_logger.info("Full Mode:{0}".format("/".join(seg_list)))

    # 精确模式
    seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
    common_logger.info("Default Mode:{0}".format("/".join(seg_list)))

    # 不使用HMM模型
    seg_list = jieba.cut("他来到了网易杭研大厦", HMM=False)
    common_logger.info("不使用HMM模型:{0}".format("/".join(seg_list)))

    # 使用HMM模型
    seg_list = jieba.cut("他来到了网易杭研大厦", HMM=True)
    common_logger.info("使用HMM模型:{0}".format("/".join(seg_list)))

    # 搜索引擎模式
    seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造", HMM=False)
    common_logger.info("搜索引擎模式:{0}".format("/".join(seg_list)))

    # 搜索引擎模式，且返回结果拼装为list，通常情况返回结果为生成器类型
    seg_list = jieba.lcut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造", HMM=True)
    common_logger.info(seg_list)


def test_user_dict():
    """
    测试自定义词典。
    :return:
    """
    test_sentences = (
        "李小福是创新办主任也是云计算方面的专家; 什么是八一双鹿\n"
        "例如我输入一个带“韩玉赏鉴”的标题，在自定义词库中也增加了此词为N类\n"
        "「台中」正確應該不會被切開。mac上可分出「石墨烯」；此時又可以分出來凯特琳了。"
    )

    # 在使用自定义词典或添加词条前的切割测试
    words = jieba.lcut(test_sentences)
    common_logger.info("原始分词结果：\n{0}".format("/".join(words)))

    # 加载用户自定义词典
    jieba.load_userdict("userdict.txt")
    words = jieba.cut(test_sentences)  # 用户自定义词典中出现的词条被正确划分
    common_logger.info("自定义词典后的分词结果：\n{0}".format("/".join(words)))

    # 添加词条石墨烯和凯特琳
    jieba.add_word("石墨烯")
    jieba.add_word("凯特琳")
    # 删除词条自定义词
    jieba.del_word("自定义词")

    words = jieba.cut(test_sentences)  # 再次进行测试
    common_logger.info("修改自定义词典后的分词结果：\n{0}".format("/".join(words)))


def test_adjust_dict():
    """
    测试调整词典中的词频。
    :return:
    """
    sentence = "如果放到post中将出错。"
    words = jieba.cut(sentence, HMM=False)
    common_logger.info("修改词频前:\n{0}".format("/".join(words)))

    jieba.suggest_freq(("中", "将"), True)
    words = jieba.cut(sentence, HMM=False)
    common_logger.info("修改词频后:\n{0}".format("/".join(words)))

    sentence = "台中正确应该不会被切开"
    words = jieba.cut(sentence, HMM=False)
    common_logger.info("修改词频前:\n{0}".format("/".join(words)))

    jieba.suggest_freq("台中", True)
    words = jieba.cut(sentence, HMM=False)
    common_logger.info("修改词频后:\n{0}".format("/".join(words)))


def test_extract_tag():
    """
    测试提取关键词。
    :return:
    """
    content = """sentence为待提取文本。
                 topK为返回几个TF-IDF权重最大的关键词，默认值为20。
                 withWeight为是否一并返回关键词权重值，默认值为FALSE。
                 allowPOS仅包括指定词性的词，默认值为空，即不筛选。
                 jieba.analyse.TFIDF(idf_path = None)新建TFIDF实例，idf_path为IDF频率文件。
            """
    words = jieba.cut(content)
    common_logger.info("分词结果:{0}".format("/".join(words)))

    tags = jieba.analyse.extract_tags(content, topK=10)
    common_logger.info("关键词提取结果:{0}".format("/".join(tags)))


def test_extract_tag_with_weight():
    """
    测试基于权重的关键词提取。
    :return:
    """
    content = "此外，公司拟对全资子公司吉林欧亚置业有限公司增资4.3亿元，" \
              "增资后，吉林欧亚置业注册资本由7000万元增加到5亿元。" \
              "吉林欧亚置业主要经营范围为房地产开发及百货零售等业务。" \
              "目前在建吉林欧亚城市商业综合体项目。2013年，实现营业收入0万元，实现净利润-139.13万元。"
    common_logger.info("基于权重的关键词提取")
    for x, weight in jieba.analyse.extract_tags(content, withWeight=True):
        common_logger.info(("{0} {1}".format(x, weight)))

    line_seg = jieba.cut(content)
    for line in line_seg:
        common_logger.info(line)


def test_textrank():
    """
    测试textrank方法，textrank是一种关键词提取的方法。
    :return:
    """
    sentence = "此外，公司拟对全资子公司吉林欧亚置业有限公司增资4.3亿元，" \
               "增资后，吉林欧亚置业注册资本由7000万元增加到5亿元。" \
               "吉林欧亚置业主要经营范围为房地产开发及百货零售等业务。目前在建吉林欧亚城市商业综合体项目。" \
               "2013年，实现营业收入0万元，实现净利润-139.13万元。"
    common_logger.info("TextRank关键词提取")
    for tag in jieba.analyse.textrank(sentence, withWeight=True):
        common_logger.info(tag)


def test_posseg():
    """
    测试词性标注。
    :return:
    """
    words = posseg.cut("我爱北京天安门，数字是123。")
    for word, flag in words:
        common_logger.info("{0} {1}".format(word, flag))


def test_tokenize():
    """
    测试token。
    :return:
    """
    # 生成token串
    result = jieba.tokenize("永和服装饰品有限公司")
    for tk in result:
        common_logger.info("word {0}\t\t start:{1}\t\t end:{2}".format(tk[0], tk[1], tk[2]))

    # 搜索模式的token串
    result = jieba.tokenize("永和服装饰品有限公司", mode="search")
    for tk in result:
        common_logger.info("word {0}\t\t start:{1}\t\t end:{2}".format(tk[0], tk[1], tk[2]))


if __name__ == '__main__':
    # test_segment()
    # test_user_dict()
    # test_posseg()
    # test_tokenize()
    # test_textrank()
    # test_adjust_dict()
    test_extract_tag_with_weight()
    pass
