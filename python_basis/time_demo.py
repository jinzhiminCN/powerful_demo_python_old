# -*- coding:utf-8 -*-

# ==============================================================================
# 测试time和datetime等日期和时间相关的方法。
# ==============================================================================
import time
from datetime import date, timedelta, timezone
import datetime

# =========================== 全局常量 ===========================
default_time_format = '%Y-%m-%d %H:%M:%S'

# =========================== test function ===========================


def test_make_struct_time():
    """
    测试生成时间元组的方法。
    :return:
    """
    # gmtime(),localtime()和strptime()以时间元祖(struct_time)的形式返回
    epoch = time.gmtime(0)
    print("epoch struct_time: {0}".format(epoch))

    local_time = time.localtime()
    print("local now struct_time: {0}".format(local_time))

    # time.time()返回时间戳
    time_stamp = time.time()
    time_stamp_struct = time.localtime(time_stamp)
    print("time stamp {0} == local struct_time {1}".format(time_stamp, time_stamp_struct))

    # 时间字符串转换为时间元组
    time_str = '2013-10-10 23:40:00'
    struct_time = str_to_struct_time(time_str)
    print("time str {0} == struct_time {1}".format(time_str, struct_time))

    # 生成timetuple
    some_date = date(2000, 1, 1)
    time_tuple = some_date.timetuple()
    print("datetime date timetuple {0}".format(time_tuple))


def test_make_time_str():
    """
    测试生成时间字符串的方法。
    :return:
    """
    # 时间元组转换为时间字符串
    local_time = time.localtime()
    time_str = time.asctime(local_time)
    print("struct_time {0} == time str {1}".format(local_time, time_str))

    # 时间戳转换为时间字符串
    time_stamp = 1000000000
    time_str = time.ctime(time_stamp)
    print("time stamp {0} == time str {1}".format(time_stamp, time_str))

    # 时间元组转换为指定格式的时间字符串
    local_time = time.localtime()
    time_format = '%Y/%m/%d %H:%M:%S'
    time_str = time.strftime(time_format, local_time)
    print("struct_time {0} == time str {1}".format(local_time, time_str))
    print("{0}年{1}月{2}日 {3}时{4}分{5}秒".format(
        local_time.tm_year, local_time.tm_mon, local_time.tm_mday,
        local_time.tm_hour, local_time.tm_min, local_time.tm_sec))


def test_make_time_stamp():
    """
    测试生成时间戳的方法。
    :return:
    """
    # 时间元组转换为时间戳
    local_time = time.localtime()
    time_stamp = time.mktime(local_time)
    print("struct_time {0} == time stamp {1}".format(local_time, time_stamp))

    # 获取当前时间戳
    time_stamp = time.time()
    print("current time stamp {0}".format(time_stamp))

    time_str = '2013-10-10 23:40:00'
    time_stamp = str_to_time_stamp(time_str)
    print("time str {0} == time stamp {1}".format(time_str, time_stamp))


def test_datetime_date():
    """
    测试datetime.date的方法。
    :return:
    """
    # 通过年月日构造date
    test_date = date(2000, 1, 1)
    print(test_date.ctime())
    print(test_date.isocalendar())
    print(test_date.strftime(default_time_format))

    # 通过时间戳构造date
    test_date2 = date.fromtimestamp(100000000)
    print(test_date2.strftime(default_time_format))


def test_datetime_time():
    """
    测试datetime.time的方法。
    :return:
    """
    test_time = datetime.time(5, 3, 2, 1)
    print(test_time)
    tz_name = test_time.tzname()
    print(tz_name)


def test_datetime_datetime():
    """
    测试datetime.datetime的方法。
    :return:
    """
    current_time = datetime.datetime.now()
    print("current datetime {0}, type {1}".format(current_time, type(current_time)))

    dt = datetime.datetime(2015, 4, 19, 12, 20) # 用指定日期时间创建datetime
    print("construct datetime: {0}".format(dt))

    time_stamp = current_time.timestamp() # 把datetime转换为timestamp
    print("current time stamp: {0}".format(time_stamp))

    local_time = datetime.datetime.fromtimestamp(time_stamp)
    print("local time: {0}".format(local_time))  # 本地时间

    utc_time = datetime.datetime.utcfromtimestamp(time_stamp)
    print("UTC time: {0}".format(utc_time))  # UTC时间


def test_timedelta():
    """
    测试datetime的timedelta。
    :return:
    """
    now = datetime.datetime.now()
    print(now.strftime('%a, %b %d %H:%M'))

    time_after_10_hours = now + timedelta(hours=10)
    print("time_after_10_hours: {0}".format(time_after_10_hours))

    time_before_1_hours = now - timedelta(days=1)
    print("time_before_1_hours: {0}".format(time_before_1_hours))

    time_other = now + timedelta(days=2, hours=12)
    print(time_other)


def test_time_zone():
    """
    测试datetime的时区。
    :return:
    """
    # 拿到UTC时间，并强制设置时区为UTC+0:00:
    utc_dt = datetime.datetime.utcnow().replace(tzinfo=timezone.utc)
    print(utc_dt)

    # astimezone()将转换时区为北京时间:
    bj_dt = utc_dt.astimezone(timezone(timedelta(hours=8)))
    print(bj_dt)

    # astimezone()将转换时区为东京时间:
    tokyo_dt = utc_dt.astimezone(timezone(timedelta(hours=9)))
    print(tokyo_dt)

    # astimezone()将bj_dt转换时区为东京时间:
    tokyo_dt2 = bj_dt.astimezone(timezone(timedelta(hours=9)))
    print(tokyo_dt2)


def str_to_struct_time(time_str, time_format='%Y-%m-%d %H:%M:%S'):
    """
    时间字符串转换为struct_time格式
    :param time_str: 时间字符串
    :param time_format: 时间格式
    :return:
    """
    # str parse time
    return time.strptime(time_str, time_format)


def str_to_time_stamp(time_str, time_format='%Y-%m-%d %H:%M:%S'):
    """
    时间字符串转换为时间戳
    :param time_str: 时间字符串
    :param time_format: 时间格式
    :return:
    """
    struct_time = time.strptime(time_str, time_format)
    return time.mktime(struct_time)


def time_stamp_to_str(time_stamp, time_format='%Y-%m-%d %H:%M:%S'):
    """
    时间字符串转换为时间戳
    :param time_stamp: 时间字符串
    :param time_format: 时间格式
    :return:
    """
    struct_time = time.localtime(time_stamp)
    return time.strftime(time_format, struct_time)


def perform_time():
    """
    执行时间。
    :return:
    """
    print(time.perf_counter())
    time.sleep(1)
    print("sleep...1...")
    print(time.perf_counter())

    print(time.process_time())
    time.sleep(1)
    print("sleep...2...")
    print(time.process_time())

if __name__ == "__main__":
    pass
    # test_make_struct_time()
    # test_make_time_str()
    # test_make_time_stamp()
    # perform_time()
    # test_datetime_date()
    # test_datetime_time()
    # test_datetime_datetime()
    test_time_zone()

