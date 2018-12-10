# -*- coding:utf-8 -*-

# ==============================================================================
# 测试socket的相关方法。
# ==============================================================================
import socket
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()


def test_socket_attr():
    """
    测试socket的基本属性。
    :return:
    """
    hostname = socket.gethostname()
    common_logger.info("socket hostname:" + hostname)

    full_domain_name = socket.getfqdn(hostname)
    common_logger.info("fully qualified domain name:" + full_domain_name)

    host_ip = socket.gethostbyname(hostname)
    common_logger.info("host IPv4:" + host_ip)


def test_simple_socket_server():
    """
    测试简单的socket server
    :return:
    """
    socket_server = socket.socket()
    socket_server.bind(("localhost", 8080))
    socket_server.listen(5)

    conn, address = socket_server.accept()
    conn.sendall(bytes("Hello world", encoding="utf-8"))


def test_simple_socket_client():
    """
    测试简单的socket client
    :return:
    """
    socket_client = socket.socket()
    socket_client.connect(("localhost", 8080))

    ret = str(socket_client.recv(1024), encoding="utf-8")
    common_logger.info(ret)


if __name__ == "__main__":
    # test_socket_attr()
    # test_simple_socket_server()
    test_simple_socket_client()
    pass

