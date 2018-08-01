# -*- coding:utf-8 -*-

# ==============================================================================
# 测试PIL Image模块的相关方法。
# ==============================================================================
import os
from PIL import Image
import matplotlib.pyplot as plt
import math
import json
import config.common_config as com_config


resource_dir = com_config.RESOURCE_DIR
image_dir = os.path.join(resource_dir, "image_data")


# =========================== test function ===========================


def test_open_image():
    """
    打开图片并显示。
    :return:
    """
    image_file = os.path.join(image_dir, "demo1.png")
    image = Image.open(image_file)
    # print_image_pixel(image)
    image.show()


def test_new_image():
    """
    新建图片并显示。
    :return:
    """
    im = Image.new(mode="RGBA", size=(128, 128), color="#FF0000")
    # print_image_pixel(im)
    im.show()


def test_image_attribute():
    """
    显示图像属性。
    :return:
    """
    im = Image.new(mode="RGBA", size=(128, 128), color="#FF0000")

    print("图像所有模式：")
    mode_info = Image._MODEINFO
    for mode_item in mode_info.items():
        print(mode_item)

    r, g, b, alpha = im.split()
    bands = im.getbands()
    print("图像通道：{0}".format(bands))
    print("图像模式：{0}".format(im.mode))
    print("图像尺寸：{0}".format(im.size))
    print("图像拆分：{0}, {1}, {2}, {3}".format(r, g, b, alpha))

    plt.figure()
    # 原图
    plt.subplot(3, 2, 1)
    plt.imshow(im)
    # r通道
    plt.subplot(3, 2, 2)
    plt.imshow(r)
    # g通道
    plt.subplot(3, 2, 3)
    plt.imshow(g)
    # b通道
    plt.subplot(3, 2, 4)
    plt.imshow(b)
    # a通道
    plt.subplot(3, 2, 5)
    plt.imshow(alpha)
    # 显示
    plt.show()


def test_image_blend():
    """
    测试图像融合方法blend。
    使用给定的两张图像及透明度变量alpha，插值出一张新的图像。这两张图像必须有一样的尺寸和模式。
    合成公式为：out = image1 *(1.0 - alpha) + image2 * alpha
    如果变量alpha为0.0，将返回第一张图像的拷贝。如果变量alpha为1.0，将返回第二张图像的拷贝。对变量alpha的值没有限制。
    :return:
    """
    alpha = 0.3

    image_file1 = os.path.join(image_dir, "demo1.png")
    image1 = Image.open(image_file1)
    image1 = image1.convert("RGBA")
    image1_alpha = adjust_image_alpha(image1, 1 - alpha)

    image2 = Image.new(mode=image1.mode, size=image1.size, color="#FF00FF")
    image2_alpha = adjust_image_alpha(image2, alpha)

    image_blend = Image.blend(image1, image2, alpha)
    # image_blend.show()

    plt.figure()
    # 原图1
    plt.subplot(3, 2, 1)
    plt.imshow(image1)
    # 原图1—alpha
    plt.subplot(3, 2, 2)
    plt.imshow(image1_alpha)
    # 原图2
    plt.subplot(3, 2, 3)
    plt.imshow(image2)
    # 原图2-alpha
    plt.subplot(3, 2, 4)
    plt.imshow(image2_alpha)
    # 融合图片
    plt.subplot(3, 2, 5)
    plt.imshow(image_blend)
    # 显示
    plt.show()


def test_image_composite():
    """
    测试图像合成方法composite。
    使用给定的两张图像及mask图像作为透明度，插值出一张新的图像。
    变量mask图像的模式可以为“1”，“L”或者“RGBA”。所有图像必须有相同的尺寸。
    :return:
    """
    width, height = 500, 500
    size = (width, height)
    image_file1 = os.path.join(image_dir, "demo1.png")
    image1 = Image.open(image_file1)
    image1 = image1.convert("RGBA")
    image1 = image1.resize(size)

    image_file2 = os.path.join(image_dir, "753322530.jpg")
    image2 = Image.open(image_file2)
    image2 = image2.convert("RGBA")
    image2 = image2.resize(size)

    image_file3 = os.path.join(image_dir, "2254599360.jpg")
    image3 = Image.open(image_file3)
    image3 = image3.convert("RGBA")
    image3 = image3.resize(size)

    image_composite = Image.composite(image1, image2, image3)
    image_composite.show()


def print_image_pixel(image):
    """
    打印图像中的所有像素值。
    :param image
    :return:
    """
    width, height = image.size[0], image.size[1]
    for h in range(0, height):
        for w in range(0, width):
            pixel = image.getpixel((w,h))
            print("{}".format(pixel), end=" ")
        print("")


def adjust_image_alpha(image, alpha):
    """
    调整图像的alpha值。
    :param image: 原始图像
    :param alpha:
    :return:
    """
    image = image.convert('RGBA')
    im_r, im_g, im_b, im_alpha = image.split()

    width, height = image.size[0], image.size[1]
    for w in range(0, width):
        for h in range(0, height):
            pixel = im_alpha.getpixel((w, h))
            pixel = round(pixel * alpha)
            im_alpha.putpixel((w, h), pixel)

    image.putalpha(im_alpha)

    return image


if __name__ == "__main__":
    pass
    # test_open_image()
    # test_new_image()
    # test_image_attribute()
    # test_image_blend()
    test_image_composite()

