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
    测试轮廓，选择灰度范围的显示。
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
    形状转换，剪裁，上下翻转，旋转。
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


def test_filter():
    """
    模糊/平滑。
    :return:
    """
    lena_img = misc.imread(os.path.join(image_dir, "lena_gray.jpg"))
    # print_image_pixel(lena_img)
    # lena_img = Image.fromarray(lena_img)
    # lena_img.show()
    # 高斯滤镜
    blurred_lena_none = ndimage.gaussian_filter(lena_img, sigma=0)
    blurred_lena = ndimage.gaussian_filter(lena_img, sigma=3)
    very_blurred = ndimage.gaussian_filter(lena_img, sigma=5)
    # 均匀滤镜
    local_mean = ndimage.uniform_filter(lena_img, size=11)
    # 锐化处理，通过增加拉普拉斯近似增加边缘权重
    blurred_l = ndimage.gaussian_filter(lena_img, 3)
    filter_blurred_l = ndimage.gaussian_filter(blurred_l, 1)
    alpha = 30
    sharpened = blurred_l + alpha * (blurred_l - filter_blurred_l)
    # 消噪处理
    noisy = lena_img + 0.4 * lena_img.std() * np.random.random(lena_img.shape)
    gauss_denoised = ndimage.gaussian_filter(noisy, 2)
    med_denoised = ndimage.median_filter(noisy, 3)

    # 显示图像
    image_arrays = [lena_img, blurred_lena_none, blurred_lena, very_blurred,
                    local_mean, sharpened, noisy, gauss_denoised, med_denoised]
    images = ndarray_2_image(image_arrays)
    pil_image_demo.plt_images(images, 3)


def bin2gray(x):
    """
    二值化转换为灰度图。直接将1乘以255，0依旧为0，如果是其他数值，先转换为0或1。
    :param x:
    :return:
    """
    if isinstance(x, (int, float)):
        if x <= 0:
            x = 0
        else:
            x = 1
        return x * 255
    elif isinstance(x, list):
        return [bin2gray(val) for val in x]
    elif isinstance(x, np.ndarray):
        return x * 255
    else:
        return x * 255


def test_morphology():
    """
    测试图像的数学形态学。
    :return:
    """
    show_content = False
    show_images = False

    el = ndimage.generate_binary_structure(2, 1)
    el_int = el.astype(np.int)
    if show_content:
        print(el)
        print(el_int)

    # 腐蚀：最小化滤镜
    a = np.zeros((7, 7), dtype=np.int)
    a[1:6, 2:5] = 1
    a_erosion1 = ndimage.binary_erosion(a).astype(a.dtype)
    a_erosion2 = ndimage.binary_erosion(a, structure=np.ones((5, 5))).astype(a.dtype)
    if show_content:
        print(a)
        print(a_erosion1)
        print(a_erosion2)
    if show_images:
        image_arrays = [a, a_erosion1, a_erosion2]
        image_arrays = bin2gray(image_arrays)
        images = ndarray_2_image(image_arrays)
        pil_image_demo.plt_images(images, 3)

    # 膨胀：最大化滤镜
    a = np.zeros((5, 5))
    a[2, 2] = 1
    a_dilation = ndimage.binary_dilation(a).astype(a.dtype)
    if show_content:
        print(a)
        print(a_dilation)
    if show_images:
        image_arrays = [a, a_dilation]
        image_arrays = bin2gray(image_arrays)
        images = ndarray_2_image(image_arrays)
        pil_image_demo.plt_images(images, 3)

    # 开操作：腐蚀 + 膨胀
    square = np.zeros((32, 32))
    square[10:-10, 10:-10] = 1
    np.random.seed(2)
    x, y = (32 * np.random.random((2, 20))).astype(np.int)
    square[x, y] = 1

    # 图像开操作
    open_square = ndimage.binary_opening(square)
    # 先腐蚀后膨胀
    eroded_square = ndimage.binary_erosion(square)
    reconstruction = ndimage.binary_propagation(eroded_square, mask=square)

    if show_images:
        image_arrays = [square, open_square, eroded_square, reconstruction]
        image_arrays = bin2gray(image_arrays)
        images = ndarray_2_image(image_arrays)
        pil_image_demo.plt_images(images, 2)


def gray_dilation():
    """
    灰度图的形态修改
    :return:
    """
    # 灰度值图像
    im = np.zeros((64, 64))
    np.random.seed(2)
    x, y = (63 * np.random.random((2, 8))).astype(np.int)
    im[x, y] = np.arange(8)
    # print_image_pixel(im)
    # 灰度膨胀
    bigger_points = ndimage.grey_dilation(im, size=(5, 5), structure=np.ones((5, 5)))
    # print_image_pixel(bigger_points)

    square = np.zeros((16, 16))
    square[4:-4, 4:-4] = 1
    dist = ndimage.distance_transform_bf(square)
    dilate_dist = ndimage.grey_dilation(dist, size=(3, 3), structure=np.ones((3, 3)))

    images = [im, bigger_points, square, dist, dilate_dist]
    pil_image_demo.plt_images(images, 3)

    plt.figure(figsize=(12.5, 3))
    plt.subplot(141)
    plt.imshow(im, interpolation='nearest')
    plt.axis('off')
    plt.subplot(142)
    plt.imshow(bigger_points, interpolation='nearest')
    plt.axis('off')
    plt.subplot(143)
    plt.imshow(dist, interpolation='nearest')
    plt.axis('off')
    plt.subplot(144)
    plt.imshow(dilate_dist, interpolation='nearest')
    plt.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0.02, top=0.99, bottom=0.01, left=0.01, right=0.99)
    plt.show()


def edge_detect():
    """
    边缘检测
    :return:
    """
    # 合成数据
    im = np.zeros((256, 256))
    im[64:-64, 64:-64] = 1
    im = ndimage.rotate(im, 15, mode='constant')
    im_gf = ndimage.gaussian_filter(im, 8)

    # sobel检测
    sx = ndimage.sobel(im_gf, axis=0, mode='constant')
    sy = ndimage.sobel(im_gf, axis=1, mode='constant')
    sob = np.hypot(sx, sy)

    images = [im, im_gf, sx, sy, sob]
    pil_image_demo.plt_images(images, 3)


def ndarray_2_image(image_arrays):
    """
    修改图像类型为Image类型。
    :param image_arrays:
    :return:
    """
    images = []
    for image_array in image_arrays:
        # if not type(image_array) == Image.Image:
        if not isinstance(image_array, Image.Image):
            image = Image.fromarray(image_array)
            images.append(image)
        else:
            images.append(image_array)
    return images


def print_image_pixel(image):
    """
    打印图像中的所有像素值。
    :param image
    :return:
    """
    width, height = image.shape[0], image.shape[1]
    for h in range(0, height):
        for w in range(0, width):
            pixel = image[w, h]
            if isinstance(pixel, (int, float)):
                print("{0: >3}".format(pixel), end=" ")
            else:
                print("{0}".format(pixel), end=" ")
        print("")


def test_histogram():
    """
    测试直方图。
    :return:
    """
    # 构造图像
    n = 10
    length = 256
    np.random.seed(1)
    im = np.zeros((length, length))
    points = length * np.random.random((2, n ** 2))
    im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    im = ndimage.gaussian_filter(im, sigma=length / (4. * n))

    mask = (im > im.mean()).astype(np.float)
    mask += 0.1 * im
    img = mask + 0.2 * np.random.randn(*mask.shape)
    print("max:{0},min:{1}".format(np.max(img), np.min(img)))

    hist, bin_edges = np.histogram(img, bins=60)
    print("hist:{0}".format(hist))
    print("bin_edges:{0}".format(bin_edges))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    binary_img = img > 0.5
    images = [im, mask, binary_img]
    pil_image_demo.plt_images(images, 3)

    # 图像直方图
    plt.figure(figsize=(11, 4))
    plt.subplot(111)
    plt.plot(bin_centers, hist, lw=2)
    plt.axvline(0.5, color='r', ls='--', lw=2)
    plt.text(0.57, 0.8, 'histogram', fontsize=20, transform=plt.gca().transAxes)
    plt.yticks([])
    plt.subplots_adjust(wspace=0.02, hspace=0.3, top=1, bottom=0.1, left=0, right=1)
    plt.show()


def test_measurement():
    """
    测试测量对象属性。
    :return:
    """
    # 合成图像
    n = 10
    length = 256
    im = np.zeros((length, length))
    points = length * np.random.random((2, n ** 2))
    im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    im = ndimage.gaussian_filter(im, sigma=length / (4. * n))
    mask = im > im.mean()

    # 标记连接成分
    label_im, nb_labels = ndimage.label(mask)
    print("区域数量：{0}".format(nb_labels))

    # 计算每个区域的尺寸，均值等等
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    mean_vals = ndimage.sum(im, label_im, range(1, nb_labels + 1))
    print("各个区域的尺寸：{0}".format(sizes))
    print("各个区域的平均尺寸：{0}".format(mean_vals))

    # 计算尺寸小的连接成分
    mask_size = sizes < 1000
    remove_pixel = mask_size[label_im]
    print(remove_pixel.shape)
    label_im2 = label_im.copy()
    label_im2[remove_pixel] = 0

    # 使用np.searchsorted重新分配标签
    labels = np.unique(label_im2)
    label_im3 = np.searchsorted(labels, label_im2)

    # 找到关注的封闭对象区域
    slice_x, slice_y = ndimage.find_objects(label_im3 == 4)[0]
    roi = im[slice_x, slice_y]

    images = [im, mask, label_im, label_im2, label_im3, roi]
    pil_image_demo.plt_images(images, 3)


if __name__ == "__main__":
    # test_show_image()
    # save_image()
    # test_contour()
    # transform_shape()
    # test_filter()
    # test_morphology()
    # edge_detect()
    # test_histogram()
    test_measurement()
    pass

