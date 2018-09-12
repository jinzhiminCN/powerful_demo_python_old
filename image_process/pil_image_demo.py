# -*- coding:utf-8 -*-

# ==============================================================================
# 测试PIL Image模块的相关方法。
# ==============================================================================
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import matplotlib.pyplot as plt
import config.common_config as com_config
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()

resource_dir = com_config.RESOURCE_DIR
image_dir = os.path.join(resource_dir, "image_data")
FONT = 'C:/Windows/Fonts/Arial.ttf'

# =========================== function ===========================


def test_open_image():
    """
    打开图片并显示。
    :return:
    """
    # file_name = "china_mobile.tif"
    file_name = "lena.jpg"
    # file_name = "demo1.png"
    image_file = os.path.join(image_dir, file_name)
    image = Image.open(image_file)
    # print_image_pixel(image)
    image.show()
    print_image_attribute(image)
    print_image_pixel(image)


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

    images = [im, r, g, b, alpha]
    plt_images(images, 2)


def print_image_attribute(image):
    """
    显示图像的属性。
    :param image:
    :return:
    """
    bands = image.getbands()
    print("图像通道：{0}".format(bands))
    print("图像模式：{0}".format(image.mode))
    print("图像尺寸：{0}".format(image.size))


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

    images = [image1, image1_alpha, image2, image2_alpha, image_blend]
    plt_images(images, 2)


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

    im_r, im_g, im_b, im_alpha = image1.split()
    print("im_g mode:{0}".format(im_g.mode))

    image_composite = Image.composite(image1, image3, im_g)
    image_composite.show()

    images = [image1, image2, image3, image_composite, im_g]
    plt_images(images, 2)


def test_image_merge():
    """
    测试图像方法merge。
    使用一些单通道图像，创建一个新的图像。变量bands为一个图像的元组或者列表，每个通道的模式由变量mode描述。
    所有通道必须有相同的尺寸。
    :return:
    """
    width, height = 500, 500
    size = (width, height)
    image_file1 = os.path.join(image_dir, "demo1.png")
    image1 = Image.open(image_file1)
    image1 = image1.convert("RGBA")
    image1 = image1.resize(size)
    r1, g1, b1, alpha1 = image1.split()

    image_file2 = os.path.join(image_dir, "753322530.jpg")
    image2 = Image.open(image_file2)
    image2 = image2.convert("RGBA")
    image2 = image2.resize(size)
    r2, g2, b2, alpha2 = image2.split()

    image_file3 = os.path.join(image_dir, "2254599360.jpg")
    image3 = Image.open(image_file3)
    image3 = image3.convert("RGBA")
    image3 = image3.resize(size)
    r3, g3, b3, alpha3 = image3.split()

    image_bands = [r1, g2, b3, alpha3]

    image_merge = Image.merge("RGBA", image_bands)
    image_merge.show()

    images = [image1, image2, image3, r1, g2, b3, alpha3, image_merge]
    plt_images(images, 3)


def print_image_pixel(image):
    """
    打印图像中的所有像素值。
    :param image
    :return:
    """
    width, height = image.size[0], image.size[1]
    for h in range(0, height):
        for w in range(0, width):
            pixel = image.getpixel((w, h))
            if isinstance(pixel, (int, float)):
                print("{0: >3}".format(pixel), end=" ")
            else:
                print("{0}".format(pixel), end=" ")
        print("")


def adjust_image_alpha(image, alpha):
    """
    调整图像的alpha值，原始值乘以alpha。
    :param image: 原始图像
    :param alpha: 透明度alpha值
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


def plt_images(images, line_size=2, interpolation=None):
    """
    使用plt显示图片。
    :param images: list列表形式的image
    :param line_size: 每行显示数量
    :param interpolation: 插值算法
    :return:
    """
    line_num = len(images)//line_size
    line_num += 1

    plt.figure()
    for i in range(0, len(images)):
        plt.subplot(line_num, line_size, i+1)
        plt.imshow(images[i], interpolation=interpolation)
        plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0.02, top=0.99, bottom=0.01, left=0.01, right=0.99)
    plt.show()


def test_image_eval():
    """
    Image.eval(image,function) ⇒ image
    使用变量function对应的函数（该函数应该有一个参数）处理变量image所代表图像中的每一个像素点。
    如果变量image所代表图像有多个通道，那变量function对应的函数作用于每一个通道。
    注意：变量function对每个像素只处理一次，所以不能使用随机组件和其他生成器。
    :return:
    """
    image_file1 = os.path.join(image_dir, "demo1.png")
    image1 = Image.open(image_file1)
    print("png mode:{0}".format(image1.mode))

    image2 = Image.eval(image1, image_fun)

    images = [image1, image2]
    plt_images(images)


def image_fun(pixel):
    """
    操作每个像素值。
    :param pixel
    :return:
    """
    # print(pixel)
    return pixel*0.5


def test_convert_gray():
    """
    转换为灰度图。
    :return:
    """
    image_file = os.path.join(image_dir, "demo1.png")
    image = Image.open(image_file)
    image_gray = image.convert("L")
    image_binary = image.convert("1")
    image_cmyk = image.convert("CMYK")

    images = [image, image_cmyk, image_gray, image_binary]
    plt_images(images)


def test_crop_image():
    """
    测试剪切图像。
    :return:
    """
    image_file = os.path.join(image_dir, "demo1.png")
    image_file2 = os.path.join(image_dir, "demo2.png")
    image = Image.open(image_file)
    width, height = image.size
    box = (0, 0, width*0.8, height*0.8)
    # 剪切图像
    image_crop = image.crop(box)
    image_crop.show()
    # # 保存图像
    # image_crop.save(image_file2)


def test_thumbnail():
    """
    测试图像变成缩略图标。
    :return:
    """
    image_file = os.path.join(image_dir, "demo1.png")
    image = Image.open(image_file)

    # 变成缩略图，等比例缩放，按照高和宽中较大值来计算比例
    image.thumbnail((128, 128))
    image.show()
    common_logger.info("thumbnail size:{0}".format(image.size))


def add_watermark(in_file, text, out_file='watermark.jpg', angle=23, opacity=0.25):
    """
    添加水印。
    https://www.pythoncentral.io/watermark-images-python-2x/
    :param in_file: 输入图片
    :param text: 水印文本
    :param out_file: 输出图片
    :param angle: 角度
    :param opacity: 透明度
    :return:
    """
    img = Image.open(in_file).convert('RGB')
    watermark = Image.new('RGBA', img.size, (0, 0, 0, 0))

    # 找到最大字体
    size = 2
    n_font = ImageFont.truetype(FONT, size)
    n_width, n_height = n_font.getsize(text)
    while n_width + n_height < watermark.size[0]:
        size += 2
        n_font = ImageFont.truetype(FONT, size)
        n_width, n_height = n_font.getsize(text)

    # 构造水印图像
    draw = ImageDraw.Draw(watermark, 'RGBA')
    draw.text(((watermark.size[0] - n_width) / 2,
               (watermark.size[1] - n_height) / 2),
              text, font=n_font)
    watermark = watermark.rotate(angle, Image.BICUBIC)
    alpha = watermark.split()[3]
    alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
    watermark.putalpha(alpha)

    # 合成图像和水印
    Image.composite(watermark, img, watermark).save(out_file, 'JPEG')


def test_watermark():
    """
    测试构造水印。
    :return:
    """
    image_file = os.path.join(image_dir, "demo1.png")
    image_watermark_file = os.path.join(image_dir, "demo_watermark.png")
    content = "WaterMark"
    add_watermark(image_file, content, image_watermark_file)


def convert_binary_by_histogram(image, ratio=0.95):
    """
    根据直方图来进行图像二值化。
    :param image:
    :param ratio: 像素值调整比例
    :return:
    """
    height, width = image.size[:2]
    img_size = height * width

    # 获取直方图和平均像素值
    img_histogram = image.histogram()
    color_values = [x for x in range(len(img_histogram))]
    np_histogram = np.array(img_histogram)
    np_color_values = np.array(color_values)
    img_color_sum = np.sum(np_histogram * np_color_values)
    img_color_avg = img_color_sum / img_size

    # 根据平均像素值进行二值化
    img_binary = convert_binary(image, int(img_color_avg * ratio))

    return img_binary


def convert_binary(image, threshold=None):
    """
    图像二值化。
    :param image:
    :param threshold:
    :return:
    """
    width, height = image.size
    image_binary = Image.new("L", (width, height), color=255)

    if threshold is None:
        image_binary = image.convert("1")
    else:
        for i in range(width):
            for j in range(height):
                if image.getpixel((i, j)) > threshold:
                    image_binary.putpixel((i, j), 255)
                else:
                    image_binary.putpixel((i, j), 0)
    return image_binary


def convert_gray(image):
    """
    图像转换为灰度图。
    :param image:
    :return:
    """
    return image.convert("L")


def test_histogram():
    """
    测试图像的直方图。
    :return:
    """
    image_file = os.path.join(image_dir, "demo1.png")
    image = Image.open(image_file)
    image_histogram = image.histogram()
    common_logger.info(image_histogram)

    color_values = [x for x in range(len(image_histogram))]
    common_logger.info(color_values)

    plt.figure()
    plt.plot(color_values, image_histogram)
    plt.show()


def show_histogram(image):
    """
    显示图像的颜色直方图分布。
    :param image:
    :return:
    """
    image_histogram = image.histogram()
    color_values = [x for x in range(len(image_histogram))]

    # 显示图像颜色的直方图
    plt.figure()
    plt.plot(color_values, image_histogram)
    plt.show()


def image_add_horizontal(image1, image2, blank_length=0):
    """
    将两张图片横向拼接。
    :param image1: Image类型的图像
    :param image2: Image类型的图像
    :param blank_length: 图像间嵌入空白的长度
    :return: 返回拼接成功的图像
    """
    if blank_length < 0:
        blank_length = 0

    width1, height1 = image1.size[0], image1.size[1]
    width2, height2 = image2.size[0], image2.size[1]
    width = width1 + width2 + blank_length
    height = max(height1, height2)

    image = Image.new("RGBA", (width, height), (255, 255, 255))
    image.paste(image1, box=(0, 0, width1, height1))
    image.paste(image2, box=(width1 + blank_length, 0, width, height2))

    return image


def image_add_vertical(image1, image2):
    """
    将两张图片纵向拼接。
    :param image1: Image类型的图像
    :param image2: Image类型的图像
    :return: 返回拼接成功的图像
    """
    width1, height1 = image1.size[0], image1.size[1]
    width2, height2 = image2.size[0], image2.size[1]
    width = max(width1, width2)
    height = height1 + height2

    image = Image.new("RGBA", (width, height), (255, 255, 255))
    image.paste(image1, box=(0, 0, width1, height1))
    image.paste(image2, box=(0, height1, width2, height))

    return image


if __name__ == "__main__":
    pass
    # test_open_image()
    # test_convert_gray()
    # test_new_image()
    # test_image_attribute()
    # test_image_blend()
    # test_image_composite()
    # test_image_eval()
    # test_image_merge()
    # test_crop_image()
    # test_thumbnail()
    # test_watermark()
    test_histogram()

