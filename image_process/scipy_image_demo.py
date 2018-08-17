# -*- coding:utf-8 -*-

# ==============================================================================
# 测试scipy中图像的相关方法。
# ==============================================================================
import scipy
import os
import imageio
from PIL import Image
from scipy import misc
from scipy import ndimage
import numpy as np

import matplotlib.pyplot as plt
import config.common_config as com_config
from image_process import pil_image_demo

resource_dir = com_config.RESOURCE_DIR
image_dir = os.path.join(resource_dir, "image_data")
lena_img_path = os.path.join(image_dir, "lena.jpg")


def test_show_image():
    """
    显示图像。
    :return:
    """
    lena_img = imageio.imread(lena_img_path)
    print("image type:{0}".format(type(lena_img)))
    print("image shape:{0}".format(lena_img.shape))
    print("image dtype:{0}".format(lena_img.dtype))
    print("image data:{0}".format(lena_img))
    plt.imshow(lena_img)
    plt.show()

    image = Image.fromarray(lena_img)
    image.show()


def test_save_image():
    """
    保存图像。
    :return:
    """
    lena_save_path = os.path.join(image_dir, "lena_save.jpg")
    # lena_img = misc.imread(lena_img_path)
    lena_img = imageio.imread(lena_img_path)
    imageio.imsave(lena_save_path, lena_img)


def test_contour():
    """
    测试轮廓。
    :return:
    """
    lena_img = imageio.imread(os.path.join(image_dir, "lena_gray.jpg"))
    print("image type:{0}".format(type(lena_img)))
    print("image shape:{0}".format(lena_img.shape))
    print("image dtype:{0}".format(lena_img.dtype))
    plt.contour(lena_img, [60, 200])
    plt.imshow(lena_img)
    plt.show()


def transform_shape():
    """
    形状转换。
    :return:
    """
    lena_img = imageio.imread(os.path.join(image_dir, "lena_gray.jpg"))
    width, height = lena_img.shape
    print("width:{0}, height:{1}".format(width, height))
    # Cropping
    crop_lena = lena_img[round(width/4):round(-width/4), round(height/4):round(-height/4)]
    # up <-> down flip
    flip_ud_lena = np.flipud(lena_img)
    # rotation
    rotate_lena = ndimage.rotate(lena_img, 45)
    rotate_lena_noreshape = ndimage.rotate(lena_img, 45, reshape=False)

    images = [lena_img, crop_lena, flip_ud_lena, rotate_lena, rotate_lena_noreshape]
    pil_image_demo.plt_images(images)

if __name__ == "__main__":
    # test_show_image()
    # save_image()
    # test_contour()
    transform_shape()
    pass

