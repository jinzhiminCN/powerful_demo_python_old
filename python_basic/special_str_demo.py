# -*- coding:utf-8 -*-

# ==============================================================================
# 特殊字符串的使用示例。
# MAC(Message Authentication Code，消息认证码算法)是含有密钥的散列函数算法，
# 兼容了MD和SHA算法的特性，并在此基础上加入了密钥。常把MAC称为HMAC(keyed-Hash Message Authentication Code)。
# MAC算法主要集合了MD和SHA两大系列消息摘要算法:
#     MD系列的算法有HmacMD2、HmacMD4、HmacMD5三种算法；
#     SHA系列的算法有HmacSHA1、HmacSHA224、HmacSHA256、HmacSHA384、HmacSHA512五种算法。
# ==============================================================================
import uuid
import base64
import hashlib
import hmac
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()


def test_uuid():
    """
    测试UUID的生成。
    :return:
    """
    # make a UUID based on the host ID and current time
    uuid_value1 = uuid.uuid1()
    common_logger.info(uuid_value1)

    # make a UUID using an MD5 hash of a namespace UUID and a name
    uuid_value3 = uuid.uuid3(uuid.NAMESPACE_DNS, 'python.org')
    common_logger.info(uuid_value3)

    # make a random UUID
    uuid_value4 = uuid.uuid4()
    common_logger.info(uuid_value4)

    # make a UUID using a SHA-1 hash of a namespace UUID and a name
    uuid_value5 = uuid.uuid5(uuid.NAMESPACE_DNS, 'python.org')
    common_logger.info(uuid_value5)

    # make a UUID from a string of hex digits (braces and hyphens ignored)
    uuid_value6 = uuid.UUID('{00010203-0405-0607-0809-0a0b0c0d0e0f}')
    common_logger.info("uuid Value:{0}, Type:{1}".format(uuid_value6, type(uuid_value6)))

    # get the raw 16 bytes of the UUID
    common_logger.info("uuid Bytes:{0}".format(uuid_value6.bytes))

    # make a UUID from a 16-byte string
    uuid_value7 = uuid.UUID(bytes=uuid_value6.bytes)
    common_logger.info(uuid_value7)


def test_base64():
    """
    测试base64编解码。
    :return:
    """
    en_value = base64.b64encode(b'binary\x00string')
    common_logger.info(en_value)
    de_value = base64.b64decode(b'YmluYXJ5AHN0cmluZw==')
    common_logger.info(de_value)

    # 由于标准的Base64编码后可能出现字符+和/，在URL中就不能直接作为参数，
    # 所以又有一种"url safe"的base64编码，其实就是把字符+和/分别变成-和_：
    en_value = base64.b64encode(b'i\xb7\x1d\xfb\xef\xff')
    common_logger.info(en_value)

    en_value = base64.urlsafe_b64encode(b'i\xb7\x1d\xfb\xef\xff')
    common_logger.info(en_value)

    de_value = base64.urlsafe_b64decode('abcd--__')
    common_logger.info(de_value)


def test_hashlib():
    """
    hashlib中的hash编码生成。
    :return:
    """
    md5 = hashlib.md5()
    md5.update('how to use md5 in python hashlib?'.encode('utf-8'))
    common_logger.info(md5.hexdigest())

    md5 = hashlib.md5()
    md5.update('how to use md5 in '.encode('utf-8'))
    md5.update('python hashlib?'.encode('utf-8'))
    common_logger.info(md5.hexdigest())

    sha1 = hashlib.sha1()
    sha1.update('how to use sha1 in '.encode('utf-8'))
    sha1.update('python hashlib?'.encode('utf-8'))
    common_logger.info(sha1.hexdigest())


def test_hmac():
    """
    测试hmac算法。
    :return:
    """
    message = b'Hello, world!'
    key = b'secret'
    h = hmac.new(key, message, digestmod='MD5')
    # 如果消息很长，可以多次调用h.update(msg)
    result = h.hexdigest()
    common_logger.info(result)


def hmac_md5(key, message):
    """
    使用hmac和MD5算法加密。
    :param key:
    :param message:
    :return:
    """
    return hmac.new(key.encode('utf-8'), message.encode('utf-8'), hashlib.md5).hexdigest()


if __name__ == "__main__":
    pass
    # test_uuid()
    # test_base64()
    # test_hashlib()
    test_hmac()

