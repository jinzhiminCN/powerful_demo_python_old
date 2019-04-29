# -*- coding:utf-8 -*-

# ==============================================================================
# 测试mysql的基本用法。
# https://github.com/PyMySQL/PyMySQL
# https://www.runoob.com/python3/python3-mysql.html
# ==============================================================================
import os
import pymysql

from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()

# 数据库连接信息
DB_HOST = "localhost"
DB_PORT = 3306
DB_USER = "test"
DB_PASSWD = "123456"
DB_NAME = "test"


def connect_mysql():
    """
    测试连接 mysql 数据库。
    :return:
    """
    # 打开数据库连接
    db = pymysql.Connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        passwd=DB_PASSWD,
        db=DB_NAME,
        charset='utf8')

    return db


def basic_select():
    """
    基本查询。
    :return:
    """
    db = connect_mysql()

    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()

    # 使用 execute()  方法执行 SQL 查询
    cursor.execute("SELECT VERSION()")

    # 使用 fetchone() 方法获取单条数据.
    data = cursor.fetchone()

    common_logger.info("Database version : {0} ".format(data))

    # 关闭数据库连接
    db.close()


def create_table():
    """
    创建表。
    :return:
    """
    db = connect_mysql()
    cursor = db.cursor()

    # 使用 execute() 方法执行 SQL，如果表存在则删除
    cursor.execute("DROP TABLE IF EXISTS EMPLOYEE")

    # 使用预处理语句创建表
    sql = """
        CREATE TABLE EMPLOYEE (
             FIRST_NAME  CHAR(20) NOT NULL,
             LAST_NAME  CHAR(20),
             AGE INT,
             SEX CHAR(1),
             INCOME FLOAT)
    """

    cursor.execute(sql)
    db.close()


def basic_insert():
    """
    基本的 insert 操作。
    :return:
    """
    db = connect_mysql()
    cursor = db.cursor()

    # SQL 插入语句
    sql = """
        INSERT INTO EMPLOYEE(FIRST_NAME,
             LAST_NAME, AGE, SEX, INCOME)
             VALUES ('Mac', 'Mohan', 20, 'M', 2000)
    """

    try:
        # 执行sql语句
        cursor.execute(sql)
        # 提交到数据库执行
        db.commit()
    except:
        # 如果发生错误则回滚
        db.rollback()

    db.close()


def basic_search():
    """
    基本查询操作。
    :return:
    """
    db = connect_mysql()
    cursor = db.cursor()

    # SQL 查询语句
    sql = """
        SELECT * FROM EMPLOYEE
           WHERE INCOME > %s
    """

    try:
        # 执行SQL语句
        cursor.execute(sql, 100)

        # 获取所有记录列表
        results = cursor.fetchall()
        for row in results:
            first_name = row[0]
            last_name = row[1]
            age = row[2]
            sex = row[3]
            income = row[4]

            # 打印结果
            common_logger.info("first_name={0},last_name={1},age={2},sex={3},income={4}"
                               .format(first_name, last_name, age, sex, income))
    except:
        common_logger.info("Error: unable to fetch data")

    # 关闭数据库连接
    db.close()



if __name__ == "__main__":
    # basic_select()
    # create_table()
    # basic_insert()
    basic_search()
    pass
