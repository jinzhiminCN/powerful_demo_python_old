# -*- coding:utf-8 -*-

# ==============================================================================
# 测试torndb的基本用法。
# ==============================================================================
import torndb_for_python3 as torndb
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()

host = "localhost"
database = "test"
user = "user"
password = "password"


def test_connect():
    """
    测试数据库连接。
    :return:
    """
    conn = torndb.Connection(host, database, user=user,
                             password=password, time_zone="+8:00")
    query_sql = "SELECT * FROM voice_content"
    result = conn.query(query_sql)
    common_logger.info(result)


if __name__ == "__main__":
    test_connect()
    pass

