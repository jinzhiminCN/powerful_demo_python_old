# -*- coding:utf-8 -*-

# ==============================================================================
# 测试FTP相关的使用方法。
# ==============================================================================
import ftplib
from util.log_util import LoggerUtil
import config.common_config as com_config

# 日志器
common_logger = LoggerUtil.get_common_logger()

host = com_config.FTP_HOST
username = com_config.FTP_USER
password = com_config.FTP_PASSWD
file_name = 'test.txt'


def get_ftp_client():
    """
    获取一个ftp客户端。
    :return:
    """
    # 实例化FTP对象
    ftp_client = ftplib.FTP(host)
    # 登录
    ftp_client.login(username, password)

    return ftp_client


def test_ftp_conn():
    """
    测试链接FTP服务器。
    :return:
    """
    # 实例化FTP对象
    ftp_client = ftplib.FTP(host)
    # 登录
    ftp_client.login(username, password)

    # 获取当前路径
    pwd_path = ftp_client.pwd()
    common_logger.info("FTP当前路径:{0}".format(pwd_path))

    # 逐行读取ftp文本文件
    ftp_client.retrlines('RETR {0}'.format(file_name))

    # 切换路径
    ftp_client.cwd("voice")
    pwd_path = ftp_client.pwd()
    common_logger.info("FTP当前路径:{0}".format(pwd_path))

    # 退出并关闭
    ftp_client.quit()


def test_ftp_download():
    """
    以二进制形式下载文件。
    :return:
    """
    file_remote = file_name
    file_local = 'E:\\test_data\\ftp_download.txt'

    # 设置缓冲器大小
    buf_size = 1024
    fp = open(file_local, 'wb')

    # 获取FTP客户端
    ftp_client = get_ftp_client()

    # 从文件中获取数据，写入指定文件
    ftp_client.retrbinary('RETR {0}'.format(file_remote), fp.write, buf_size)
    fp.close()


def test_ftp_upload():
    """
    以二进制形式上传文件
    :return:
    """
    file_remote = 'ftp_upload.txt'
    file_local = 'E:\\test_data\\ftp_upload.txt'
    # 设置缓冲器大小
    buf_size = 1024
    fp = open(file_local, 'rb')

    # 获取FTP客户端
    ftp_client = get_ftp_client()

    # 将文件内容写入远程文件
    ftp_client.storbinary('STOR {0}'.format(file_remote), fp, buf_size)
    fp.close()


def test_ftp_dir():
    """
    测试显示FTP的目录。
    :return:
    """
    with ftplib.FTP(host) as ftp_client:
        ftp_client.login(username, password)
        # ftp_client.dir()
        ftp_client.dir(show_list)


def show_list(content_list):
    """
    显示列表内容。
    :param:content_list内容列表
    :return:
    """
    common_logger.info(content_list)


def test_ftp_tls():
    """
    测试FTP
    :return:
    """
    ftps = ftplib.FTP_TLS('ftp.pureftpd.org')
    # 登陆
    ftps.login()

    # 设置受保护的连接
    ftps.prot_p()

    # 返回文件列表
    dir_list = ftps.nlst()
    common_logger.info(dir_list)

    # 退出
    ftps.quit()


if __name__ == "__main__":
    # test_ftp_conn()
    # test_ftp_download()
    # test_ftp_upload()
    # test_ftp_dir()
    test_ftp_tls()
    pass
