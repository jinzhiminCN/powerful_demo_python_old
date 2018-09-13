# -*- encoding:utf-8 -*-

# ==============================================================================
# 字符图像的处理。
# ==============================================================================
import os
from PIL import Image

# 定义一个ascii的列表，其实就是让图片上的灰度与字符对应
ascii_char = list("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ")


def get_char(r, g, b, alpha=256):
    """
    将256灰度映射到给出的字符上。
    :param r:
    :param g:
    :param b:
    :param alpha:
    :return:
    """
    # 如果灰度是0，说明这里没有图片
    if alpha == 0:
        return ' '
    length = len(ascii_char)  # 计算这些字符的长度

    # 把图片的RGB值转换成灰度值
    gray = int(0.2126 * r + 0.7152 * g + 0.0722 * b)
    # 每个字符对应几个灰度值
    unit = (256.0 + 1) / length

    # 这个相当于是选出了灰度与哪个字符对应
    return ascii_char[int(gray / unit)]


def image_2_char_image(image_path, char_file_path, resize=False):
    """
    普通图像转换为字符图像。
    :param image_path:
    :param char_file_path:
    :param resize: 是否缩放
    :return:
    """
    # 加载图片
    im = Image.open(image_path)
    width = im.size[0]
    height = im.size[1]

    # 为方便查看结果需要将图形缩放
    if resize:
        if width > 180 or height > 180:
            im.thumbnail((180, 180))

    # 缩放后的图像大小
    width = im.size[0]
    height = im.size[1]

    content = ""
    for i in range(height):
        for j in range(width):
            # 把图片按照横纵坐标解析成r,g,b以及alpha这几个参数，然后调用get_char函数，得到对应的字符
            content += get_char(*im.getpixel((j, i)))
        content += '\n'  # 每行的结尾处，自动换行

    char_file = open(char_file_path, 'w')
    char_file.write(content)
    char_file.close()


if __name__ == "__main__":
    dir_path = 'E:/images'
    image_name = '145234_164944034631_2.png'
    image_path = os.path.join(dir_path, image_name)
    char_file_path = os.path.join(dir_path, "{0}.txt".format(image_name[:-4]))
    image_2_char_image(image_path, char_file_path, resize=True)

