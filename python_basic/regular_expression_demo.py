# -*- coding:utf-8 -*-

# ==============================================================================
# 测试正则表达式re的相关方法。
# ==============================================================================
import re
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()


def test_findall():
    """
    测试正则表达式的findall。
    :return:
    """
    # 测试findall
    content = "<h1>hello world<h1> " \
              "<h1>second world<h1> " \
              "<hTml>hello</Html> " \
              "chuxiuhong@hit.edu.cn" \
              "saas and sas and saaas" \
              "mat cat hat pat" \
              "192.168.1.1 " \
              "127.12.33.22 " \
              "123.43.5513.23 "
    search_key_list = ["<h1>.+<h1>",
                       "<[Hh][Tt][Mm][Ll]>.+?</[Hh][Tt][Mm][Ll]>",
                       "<h1>.+?<h1>",
                       "@.+?\.",
                       "sa{1,2}s",
                       "[^p]at",
                       "(\d+{1,3}\.){3}\d+{1,3}\."]
    for search_key in search_key_list:
        re_pattern = re.compile(search_key)
        result = re_pattern.findall(content)
        common_logger.info("findall:{0}".format(result))


def test_re():
    """
    测试正则表达式的用法。
    :return:
    """
    content = "java python html css abc cdc"
    search_key_list = ["(h)", "h", "python", "(.c)"]

    # 测试分组group
    for search_key in search_key_list:
        re_pattern = re.compile(search_key)
        matcher = re.search(re_pattern, content)
        groups_content = matcher.groups()
        group_dict = matcher.groupdict()

        common_logger.info("search key:{0}".format(search_key))
        common_logger.info("group dict:{0}".format(group_dict))
        common_logger.info("groups:{0}".format(groups_content))
        common_logger.info("group :{0}".format(matcher.group()))
        display_match(matcher)

    # 子表达式
    content = "127.12.33.22 " \
              "192.168.1.1 " \
              "123.43.5513.23 "
    search_key_list = ["(\d{1,3}\.){3}\d{1,3}"]
    for search_key in search_key_list:
        re_pattern = re.compile(search_key)
        matcher = re.search(re_pattern, content)
        groups_content = matcher.groups()
        group_dict = matcher.groupdict()

        common_logger.info("search key:{0}".format(search_key))
        common_logger.info("group dict:{0}".format(group_dict))
        common_logger.info("groups:{0}".format(groups_content))
        common_logger.info("group :{0}".format(matcher.group()))
        display_match(matcher)


def test_forward_backward():
    """
    前向后向查找。
    :return:
    """
    content = "<html><body><h1>hello world</h1></body></html>"
    # 第一个?<=表示在被匹配字符前必须得有<h1>，后面的?=表示被匹配字符后必须有<h1>
    # 如果要匹配的字符是XX，但必须满足AXXB形式，正则表达式"(?<=A)XX(?=B)"
    search_key_list = ["(?<=<h1>).+?(?=</h1>)"]

    for search_key in search_key_list:
        re_pattern = re.compile(search_key)
        matcher = re.search(re_pattern, content)
        common_logger.info("group 0:{0}".format(matcher.group(0)))


def test_traceback():
    """
    回溯引用。
    :return:
    """
    content = "<h1>hello world</h3> <h2>second world</h2>"
    # \1表示第一个子表达式
    search_key_list = [r"<h([1-6])>(.*?)</h\1>", '<h([1-6])>(.*?)</h\1>']

    for search_key in search_key_list:
        re_pattern = re.compile(search_key)
        matcher = re.search(re_pattern, content)
        display_match(matcher)

    pair = re.compile(r".*(.).*\1")
    display_match(pair.match("717ak"))


def display_match(match):
    """
    显示match信息。
    :param match:
    :return:
    """
    if match is None:
        common_logger.info("None")
    else:
        result = '<Match: {0}, groups={1}>'.format(match.group(), match.groups())
        common_logger.info(result)


def test_match_search():
    """
    match和search的区别。
    :return:
    """
    common_logger.info(re.match("c", "abcdef"))
    common_logger.info(re.search("c", "abcdef"))

    common_logger.info(re.match("c", "abcdef"))
    common_logger.info(re.search("^c", "abcdef"))
    common_logger.info(re.search("^a", "abcdef"))

    common_logger.info(re.match('X', 'A\nB\nX', re.MULTILINE))
    common_logger.info(re.search('^X', 'A\nB\nX', re.MULTILINE))


def test_group():
    """
    测试分组。
    :return:
    """
    matcher = re.match(r"(?P<first_name>\w+) (?P<last_name>\w+)", "Malcolm Reynolds")
    common_logger.info("<first_name>:{0}".format(matcher.group('first_name')))
    common_logger.info("<last_name>:{0}".format(matcher.group('last_name')))
    common_logger.info("group 1:{0}".format(matcher.group(1)))
    common_logger.info("group 2:{0}".format(matcher.group(2)))


if __name__ == "__main__":
    # test_forward_backward()
    # test_traceback()
    # test_re()
    # test_match_search()
    pass
    test_group()

