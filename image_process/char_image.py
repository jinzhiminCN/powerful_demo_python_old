# -*- encoding:utf-8 -*-

# ==============================================================================
# 字符图像的处理。
# ==============================================================================
import os
from PIL import Image, ImageFont, ImageDraw
import imageio
import config.common_config as com_config
from util.os_util import OsUtil
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()

resource_dir = com_config.RESOURCE_DIR
image_dir = os.path.join(resource_dir, "image_data")

# 定义一个ascii的列表，其实就是让图片上的灰度与字符对应
ascii_char = list("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ")
# 两帧之间的时间间隔，秒为单位
DURATION = 0.1
# 字体映射
font_map = [' ', '.', 'i', 'I', 'J', 'C', 'D', 'O', 'S', 'Q', 'G', 'F', 'E', '#', '&', '@']


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


def image_2_font_image(image_path, image_save_path, max_size=100):
    """
    图像转换为字体图像。
    :param image_path:
    :param image_save_path:
    :param max_size:
    :return:
    """
    im = Image.open(image_path).convert('L')
    width = im.size[0]
    height = im.size[1]

    font_size = 16

    if width > max_size or height > max_size:
        im.thumbnail((max_size, max_size))

    # 缩放后的图像大小
    width = im.size[0]
    height = im.size[1]

    level = im.getextrema()[-1] / (len(font_map) - 1)
    im = im.point(lambda i: int(i / level))
    im_new = Image.new('L', (width * font_size, height * font_size))

    font = ImageFont.truetype('arial.ttf', font_size)
    im_draw = ImageDraw.Draw(im_new)

    for y in range(0, height):
        for x in range(0, width):
            pixel = im.getpixel((x, y))
            im_draw.text((x * font_size, y * font_size),
                         font_map[len(font_map) - pixel - 1], fill=255, font=font)

    im_new.save(image_save_path)


def test_char_image():
    """
    测试字符图像。
    :return:
    """
    dir_path = image_dir
    image_name = '145234_164944034631_2.png'
    image_path = os.path.join(dir_path, image_name)
    char_file_path = os.path.join(dir_path, "{0}.txt".format(image_name[:-4]))
    image_2_char_image(image_path, char_file_path, resize=True)


def test_font_image():
    """
    测试字体图像。
    :return:
    """
    dir_path = image_dir
    image_name = '145234_164944034631_2.png'
    image_path = os.path.join(dir_path, image_name)
    image_font_path = "{0}_font.png".format(image_path[:-4])
    image_2_font_image(image_path, image_font_path)


def gif2png(file_path):
    """
    拆分gif为png。
    :param file_path:
    :return:
    """
    file_name = os.path.basename(file_path)
    im = Image.open(file_path)

    dir_path = os.path.join(image_dir, "cache")
    OsUtil.makedirs(dir_path)
    os.chdir(dir_path)

    file_path = os.path.join(dir_path, file_name)

    try:
        while 1:
            # 获取当前帧数
            current = im.tell()
            name = file_path[:-4] + '_' + str(current) + '.png'
            # gif分割后保存的是索引颜色
            im.save(name)
            common_logger.info("png name:{0}".format(name))
            generate_txt_image(name)
            im.seek(current + 1)
    except:
        os.chdir(dir_path)


def generate_txt_image(file_name):
    """
    生成txt图片。
    先将图像转换为txt文本，再将txt转换为图片。
    :param file_name:
    :return:
    """
    # gif拆分后的图像，需要转换，否则报错，因为gif分割后保存的是索引颜色
    im = Image.open(file_name).convert('RGB')

    raw_width = im.width
    raw_height = im.height
    width = int(raw_width/6)
    height = int(raw_height/15)

    # 将图像转换为txt文本
    im = im.resize((width, height), Image.NEAREST)
    txt = ""
    colors = []
    for i in range(height):
        for j in range(width):
            pixel = im.getpixel((j, i))
            colors.append((pixel[0], pixel[1], pixel[2]))
            if len(pixel) == 4:
                txt += get_char(pixel[0], pixel[1], pixel[2], pixel[3])
            else:
                txt += get_char(pixel[0], pixel[1], pixel[2])
        txt += '\n'
        colors.append((255, 255, 255))

    # 将txt转换为图片
    im_txt = Image.new("RGB", (raw_width, raw_height), (255, 255, 255))
    dr = ImageDraw.Draw(im_txt)
    font = ImageFont.load_default().font
    x = y = 0
    # 获取字体的宽高
    font_w, font_h = font.getsize(txt[1])
    # 调整后更佳
    font_h *= 1.37
    # ImageDraw为每个ascii码进行上色
    for i in range(len(txt)):
        if txt[i] == '\n':
            x += font_h
            y = -font_w
        dr.text([y, x], txt[i], colors[i])
        y += font_w

    file_name_array = file_name.split('.')
    name = file_name_array[0] + '_txt.' + file_name_array[1]
    common_logger.info("txt image name:{0}".format(name))
    im_txt.save(name)


def png2gif(dir_name):
    """
    转换为gif。
    :param dir_name:
    :return:
    """
    path = os.getcwd()
    os.chdir(dir_name)
    dirs = os.listdir()
    images = []
    num = 0
    for image_file in dirs:
        if image_file.split('_')[-1] == 'txt.png':
            images.append(imageio.imread(image_file))
            common_logger.info(image_file)
            num += 1
    os.chdir(path)

    gif_name = image_file.split('_')[0]+'_txt_c.gif'
    common_logger.info(gif_name)
    imageio.mimsave(gif_name, images, "GIF", duration=DURATION)


def test_gif_char_image():
    """
    测试gif转换为字符图像。
    :return:
    """
    image_name = 'ce86fd8f5e8e75ab4a914c15ecd1e443.gif'
    image_path = os.path.join(image_dir, image_name)
    gif2png(image_path)
    dir_path = os.path.join(image_dir, "cache")
    png2gif(dir_path)


if __name__ == "__main__":
    # test_char_image()
    # test_gif_char_image()
    test_font_image()
    pass

