# -*- coding:utf-8 -*-

# ==============================================================================
# 测试requests的基本用法。
# https://blog.csdn.net/qq_25134989/article/details/78800209
# ==============================================================================
import requests
import re
from bs4 import BeautifulSoup
import urllib
from requests.auth import HTTPDigestAuth

from urllib import request, parse
from urllib.error import *
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()


def post_request():
    """
    测试post请求。
    :return:
    """
    url = 'http://pythonscraping.com/files/processing.php'
    params = {
        'firstname': 'Ryan',
        'lastname': 'Mitchell'
    }
    response = requests.post(url, data=params)
    common_logger.info(response.text)


def page_content():
    """
    测试页面内容。
    :return:
    """
    url = "http://www.pythonscraping.com/pages/page3.html"
    common_logger.info("**********python content***********")
    try:
        html = request.urlopen(url)
        bs_obj = BeautifulSoup(html.read(), "html.parser")

        common_logger.info(bs_obj.html)

        # 查找所有的img元素
        images = bs_obj.findAll("img", {"src": re.compile("\.\.\/img\/gifts/img.*\.jpg")})
        for image in images:
            common_logger.info(image.attrs)
            common_logger.info(image["src"])
    except HTTPError as e:
        common_logger.info(HTTPError, e)
    else:
        pass
    finally:
        pass


def download_image():
    """
    下载logo图像。
    :return:
    """
    url = "http://www.pythonscraping.com"
    html = request.urlopen(url)
    bs_obj = BeautifulSoup(html, "html.parser")

    # 解析图像url地址
    image_location = bs_obj.find("a", {"id": "logo"}).find("img")["src"]
    # 取回图像并保存
    request.urlretrieve(image_location, "logo.jpg")


def test_urlopen():
    """
    测试urlopen请求。
    :return:
    """
    with request.urlopen('https://api.douban.com/v2/book/2129650') as f:
        data = f.read()
        common_logger.info('Status:{0}, {1}'.format(f.status, f.reason))

        for k, v in f.getheaders():
            common_logger.info('Header:{0}, {1}'.format(k, v))
        common_logger.info('Data:{0}'.format(data.decode('utf-8')))


def test_phone_get():
    """
    测试手机版get请求。
    :return:
    """
    req = request.Request('http://www.douban.com/')
    req.add_header('User-Agent',
                   'Mozilla/6.0 (iPhone; CPU iPhone OS 8_0 like Mac OS X) AppleWebKit/536.26 '
                   '(KHTML, like Gecko) Version/8.0 Mobile/10A5376e Safari/8536.25')
    with request.urlopen(req) as f:
        data = f.read()
        common_logger.info('Status:{0}, {1}'.format(f.status, f.reason))

        for k, v in f.getheaders():
            common_logger.info('Header:{0}, {1}'.format(k, v))
        common_logger.info('Data:{0}'.format(data.decode('utf-8')))


def test_request_auth():
    """
    测试request的认证请求。
    :return:
    """
    response = requests.get('https://api.github.com/user', auth=('user', 'pass'))
    common_logger.info(response.text)

    url = 'http://httpbin.org/digest-auth/auth/user/pass'
    response = requests.get(url, auth=HTTPDigestAuth('user', 'pass'))
    common_logger.info(response.text)


def download_image_from_url(url, file_path):
    """
    从URL中下载文件，保存到file_path。
    :param url:下载文件的url
    :param file_path:保存图像的文件路径
    :return:
    """
    # 发送请求
    response = requests.get(url)
    common_logger.info("URL:{0}，请求状态:{1}".format(url, response.status_code))

    # 获取的文本实际上是图片的二进制文本
    img = response.content
    # 拷贝到本地文件 w 写 b 二进制，wb代表写入二进制文本
    with open(file_path, 'wb') as file:
        file.write(img)


def test_request_basic_method():
    """
    测试请求的基本方法。
    :return:
    """
    response = requests.post("http://httpbin.org/post")
    common_logger.info("content:{0}".format(response.content))
    common_logger.info("text:{0}".format(response.text))
    common_logger.info("status code:{0}".format(response.status_code))

    response = requests.put("http://httpbin.org/put")
    common_logger.info("text:{0}".format(response.text))

    response = requests.delete("http://httpbin.org/delete")
    common_logger.info("text:{0}".format(response.text))

    response = requests.head("http://httpbin.org/get")
    common_logger.info("text:{0}".format(response.text))

    response = requests.options("http://httpbin.org/get")
    common_logger.info("text:{0}".format(response.text))

    payload = {'key1': 'value1', 'key2': 'value2'}
    response = requests.get("http://httpbin.org/get", params=payload)
    common_logger.info("url:{0}".format(response.url))

    payload = {'key1': 'value1', 'key2': ['value2', 'value3']}
    response = requests.get("http://httpbin.org/get", params=payload)
    common_logger.info("url:{0}".format(response.url))

    response = requests.post("http://httpbin.org/post", data=payload)
    common_logger.info("text:{0}".format(response.text))


if __name__ == "__main__":
    # post_request()
    # page_content()
    # download_image()
    # test_urlopen()
    # test_phone_get()
    # test_request_auth()
    test_request_basic_method()
    pass
