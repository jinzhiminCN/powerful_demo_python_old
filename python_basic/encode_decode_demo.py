# -*- coding:utf-8 -*-

# ==============================================================================
# 编解码相关的方法，包括encode decode等。
# 汉字编码 https://www.qqxiuzi.cn/zh/hanzi-unicode-bianma.php
# ==============================================================================

chinese_code_dict = {
    "基本汉字": "4E00-9FA5",
    "基本汉字补充": "9FA6-9FEF",
    "扩展A": "3400-4DB5",
    "扩展B": "20000-2A6D6",
    "扩展C": "2A700-2B734",
    "扩展D": "2B740-2B81D",
    "扩展E": "2B820-2CEA1",
    "扩展F": "2CEB0-2EBE0",
    "康熙部首": "2F00-2FD5",
    "部首扩展": "2E80-2EF3",
    "兼容汉字": "F900-FAD9",
    "兼容扩展": "2F800-2FA1D",
    "PUA(GBK)部件": "E815-E86F",
    "部件扩展": "E400-E5E8",
    "PUA增补": "E600-E6CF",
    "汉字笔画": "31C0-31E3",
    "汉字结构": "2FF0-2FFB",
    "汉语注音": "3105-312F",
    "注音扩展": "31A0-31BA",
    "〇": "3007",
}


def test_encode_decode():
    """
    测试编解码相关内容。
    encode将unicode字符串转换为byte, decode将byte转换为unicode字符串。
    u.encode('...')基本上总是能成功，只要填写了正确的编码。
    s.decode('...')经常是会出错的，因为str的“编码”取决于上下文。
    :return:
    """
    # 普通定义字符串的方式
    common_str = "Hello world ２４８Ｑ字符串"

    # 直接定义unicode字符串，通过在字符串前加u的方式
    unicode_str = u"Hello world ２４８Ｑ字符串"
    print("原始unicode字符串：{0},类型：{1}".format(unicode_str, type(unicode_str)))

    unicode_byte = unicode_str.encode('unicode_escape')
    print("转换为unicode byte编码：{0},类型：{1}".format(unicode_byte, type(unicode_byte)))

    unicode_value = unicode_str.encode('unicode_escape').decode()
    print("转换为unicode编码：{0},类型：{1}".format(unicode_value, type(unicode_value)))

    utf_str = unicode_str.encode("utf-8")
    print(utf_str)
    print(utf_str.decode())

    gbk_str = unicode_str.encode("gbk")
    print(gbk_str)
    print(gbk_str.decode("gbk"))


def change_to_unicode(content):
    """
    将字符串转换为unicode编码格式。
    :return:
    """
    return content.encode("unicode_escape")


def is_chinese(uchar):
    """
    判断一个unicode是否是汉字。
    :param uchar:
    :return:
    """
    x = ord(uchar)
    # Punct & Radicals
    if 0x2e80 <= x <= 0x33ff:
        return True
    # Fullwidth Latin Characters
    elif 0xff00 <= x <= 0xffef:
        return True
    # CJK Unified Ideographs &
    # CJK Unified Ideographs Extension A
    elif 0x4e00 <= x <= 0x9fbb:
        return True
    # CJK Compatibility Ideographs
    elif 0xf900 <= x <= 0xfad9:
        return True
    # CJK Unified Ideographs Extension B
    elif 0x20000 <= x <= 0x2a6d6:
        return True
    # CJK Compatibility Supplement
    elif 0x2f800 <= x <= 0x2fa1d:
        return True
    else:
        return False


def is_number(uchar):
    """
    判断一个unicode是否是数字。
    """
    if u'\u0030' <= uchar <= u'\u0039':
        return True
    else:
        return False


def is_alphabet(uchar):
    """
    判断一个unicode是否是英文字母。
    """
    if (u'\u0041' <= uchar <= u'\u005a') or (u'\u0061' <= uchar <= u'\u007a'):
        return True
    else:
        return False


def is_other(uchar):
    """
    判断是否非汉字，数字和英文字符。
    """
    if not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar)):
        return True
    else:
        return False


def test_unicode():
    """
    测试unicode相关内容。
    :return:
    """
    common_str = "Hello world ２４８Ｑ字符串"
    for char_word in common_str:
        print("{0: >4} 中文：{1}，ord：{2}, 编码：{3}".format(
            char_word, is_chinese(char_word), ord(char_word), change_to_unicode(char_word)))

    # 使用unicode编码组成字符串
    unicode_str = u'\u31A0\u4e00\u9FA5\u3007\u3105'
    print(unicode_str)

    # 字符串全角转换半角
    print(str_full_to_half(common_str))


def half_to_full_width(uchar):
    """
    半角转全角。
    """
    inside_code = ord(uchar)
    # 不是半角字符就返回原来的字符
    if inside_code < 0x0020 or inside_code > 0x7e:
        return uchar

    # 除了空格其他的全角半角的公式为:半角=全角-0xfee0
    if inside_code == 0x0020:
        inside_code = 0x3000
    else:
        inside_code += 0xfee0
    return chr(inside_code)


def full_to_half_width(uchar):
    """
    全角转半角。
    """
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0

    # 转完之后不是半角字符返回原来的字符
    if inside_code < 0x0020 or inside_code > 0x7e:
        return uchar
    return chr(inside_code)


def str_full_to_half(ustring):
    """
    把字符串全角转半角
    """
    return "".join([full_to_half_width(uchar) for uchar in ustring])


def uniform(ustring):
    """
    格式化字符串，完成全角转半角，大写转小写的工作
    """
    return str_full_to_half(ustring).lower()


if __name__ == "__main__":
    pass
    test_encode_decode()
    # print(change_to_unicode("ABC --- 哲学"))
    # test_unicode()


