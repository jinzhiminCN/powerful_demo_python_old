# -*- coding:utf-8 -*-

# ==============================================================================
# 测试使用json的各种方法。
# JSON (JavaScript Object Notation)
# https://docs.python.org/3/library/json.html
# ==============================================================================
import json
import datetime
import time
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()


class DateEncoder(json.JSONEncoder):
    """
    json日期编码类。
    """
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, datetime.date):
            return obj.strftime("%Y-%m-%d")
        else:
            return json.JSONEncoder.default(self, obj)


def test_json_dump():
    """
    测试json的dump操作。将json对象转换为json字符串。dump相当于存储起来。
    :return:
    """
    json_obj1 = [[1, 2, 3], 123, 123.123, 'abc', {'key1': (1, 2, 3), 'key2': (4, 5, 6)}]
    json_str = json.dumps(json_obj1)

    common_logger.info("json repr:{0}".format(repr(json_str)))
    common_logger.info("json type:{0}".format(str(type(json_str))))
    common_logger.info("json str:{0}".format(json_str))
    for json_item in json_obj1:
        common_logger.info("json obj item value:{0}, type:{1}".format(json_item, str(type(json_item))))


def test_json_load():
    """
    测试json的load操作。将json字符串转换为json对象。
    :return:
    """
    json_str = '{"name":"mike", "age":"12"}'
    json_obj = json.loads(json_str)

    if 'name2' in json_obj:
        common_logger.info("json obj has name2")
    else:
        common_logger.info("json obj not has name2")

    common_logger.info("json obj element name:{0}".format(json_obj.get("name")))
    common_logger.info("json obj element name:{0}".format(json_obj.get("name2")))
    # common_logger.info("json obj element name:{0}".format(json_obj["name2"]))
    common_logger.info("json obj type:{0}".format(str(type(json_obj))))


def test_json_date_time():
    """
    测试json中的日期和时间。
    :return:
    """
    # 普通字符串
    json_obj1 = [[1, 2, 3], 'abc', '2018-01-02', {'key1': (1, 2, 3), 'key2': (4, 5, 6)}]
    json_str1 = json.dumps(json_obj1)
    common_logger.info("json str:{0}".format(json_str1))

    # time_struct类型的时间
    json_obj2 = [[1, 2, 3], 'abc', time.localtime(), {'key1': (1, 2, 3), 'key2': (4, 5, 6)}]
    common_logger.info("json obj:{0}".format(json_obj2))
    json_str2 = json.dumps(json_obj2)
    common_logger.info("json str:{0}".format(json_str2))
    json_obj2 = json.loads(json_str2)
    common_logger.info("json obj after:{0}".format(json_obj2))

    # datetime类型的时间
    json_obj3 = [[1, 2, 3], 'abc', datetime.datetime.now(), {'key1': (1, 2, 3), 'key2': (4, 5, 6)}]
    common_logger.info("json obj:{0}".format(json_obj3))
    try:
        json_str3 = json.dumps(json_obj3)
        common_logger.info("json str:{0}".format(json_str3))
    except TypeError as err:
        common_logger.info("json dump error:{0}".format(err))
    json_str3 = json.dumps(json_obj3, cls=DateEncoder)
    common_logger.info("json str:{0}".format(json_str3))
    json_obj3 = json.loads(json_str3)
    common_logger.info("json obj after:{0}".format(json_obj3))


if __name__ == '__main__':
    pass
    # test_json_dump()
    test_json_load()
    # test_json_date_time()

