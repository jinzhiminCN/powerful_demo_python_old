# -*- coding:utf-8 -*-

# ==============================================================================
# 测试requests的基本用法。
# ==============================================================================
import requests
import re
from bs4 import BeautifulSoup
import urllib
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


if __name__ == "__main__":
    # post_request()
    # page_content()
    # download_image()
    # test_urlopen()
    test_phone_get()
    pass
