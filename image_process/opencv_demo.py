# -*- coding:utf-8 -*-

# ==============================================================================
# 测试OpenCV cv2模块的相关方法。
# ==============================================================================
import cv2
import numpy as np
import os
import time
from image_process import pil_image_demo
import config.common_config as com_config


resource_dir = com_config.RESOURCE_DIR
image_dir = os.path.join(resource_dir, "image_data")

# =========================== function ===========================


def test_show_image():
    """
    读取、显示以及保存图片。
    :return:
    """
    image_file1 = os.path.join(image_dir, "demo1.png")
    image_file2 = os.path.join(image_dir, "test2.jpg")

    # 以灰度图形式读取一张图片
    img = cv2.imread(image_file1, cv2.IMREAD_GRAYSCALE)

    # 创建一个名为Example的窗口
    cv2.namedWindow("Example", cv2.WINDOW_NORMAL)

    # 在窗口中显示这张图片
    cv2.imshow("Example", img)

    # 等待回应
    k = cv2.waitKey()
    if k == 27:  # 等待Esc退出
        cv2.destroyAllWindows()
    elif k == ord('s'):  # 等待s键保存
        cv2.imwrite(image_file2, img)
        cv2.destroyAllWindows()


def plot_images():
    """
    用opencv作图，包括直线、圆、填充字符。
    :return:
    """
    img = np.zeros((512, 512, 3), np.uint8)

    # 画一条线，参数为图片，起点，终点，颜色，线条类型
    cv2.line(img, (0, 0), (512, 512), (255, 0, 0), 5)

    # 画矩形，参数为图片，左上角顶点，右下角顶点，颜色，线条类型
    cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)

    # 画圆，参数为图片，中心点坐标，半径，颜色，线条类型：填充
    cv2.circle(img, (447, 63), 63, (0, 0, 255), -1)

    # 画椭圆，参数为图片，中心点坐标，长短轴，逆时针旋转的角度，
    # 椭圆弧沿顺时针方向的起始角度和结束角度，颜色类型填充
    cv2.ellipse(img, (256, 256), (100, 50), 0, 0, 180, 255, -1)

    # pts = np.array([[10,5],[20,30],[70,20],[50,10]],np.int32)
    # pts = pts.reshape((-1,1,2))

    # 在图片添加文字，参数为，图片，绘制文字，位置，字体类型，字体大小，颜色，线条类型
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 2)

    show_image(img)


def show_image(img):
    """
    显示图像。
    :param img:
    :return:
    """
    win_name = 'example'
    cv2.namedWindow(win_name)
    cv2.imshow(win_name, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def convert_image_color():
    """
    图像颜色转换。
    :return:
    """
    image_file1 = os.path.join(image_dir, "demo1.png")
    image1 = cv2.imread(image_file1)
    image_cvt1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    image_cvt2 = cv2.cvtColor(image1, cv2.COLOR_RGB2HSV)
    image_cvt3 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image_cvt4 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image_cvt5 = cv2.cvtColor(image_cvt3, cv2.COLOR_RGB2GRAY)

    images = [image1, image_cvt1, image_cvt2, image_cvt3, image_cvt4, image_cvt5]
    pil_image_demo.plt_images(images)


def test_image_resize():
    """
    测试图像缩放。
    :return:
    """
    image_file = os.path.join(image_dir, "demo1.png")
    image = cv2.imread(image_file)
    print("图像的类型：{0}".format(type(image)))
    print("size {0}, shape {1}".format(image_file, image.size, image.shape))

    # 图像缩放
    img_resize1 = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    print("size {0}, shape {1}".format(img_resize1.size, img_resize1.shape))
    show_image(img_resize1)

    img_resize2 = cv2.resize(image, (300, 200), interpolation=cv2.INTER_AREA)
    print("size {0}, shape {1}".format(img_resize2.size, img_resize2.shape))
    show_image(img_resize2)


def test_image_cut_makeborder():
    """
    图像裁剪和补边，裁剪是利用array自身的下标截取实现，此外OpenCV还可以给图像补边。
    :return:
    """
    image_file = os.path.join(image_dir, "demo1.png")
    img = cv2.imread(image_file)

    img_cut = img[30:180, 40:150]
    img_border = cv2.copyMakeBorder(img, 10, 10, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    show_image(img_cut)
    show_image(img_border)


def test_image_rotate():
    """
    测试图像旋转。
    在OpenCV中，仿射变换的矩阵是一个2×3的矩阵，其中左边的2×2子矩阵是线性变换矩阵，右边的2×1的两项是平移项。
    :return:
    """
    image_file = os.path.join(image_dir, "demo1.png")

    # 图像读取模式
    iread_mode = cv2.IMREAD_ANYCOLOR
    img = cv2.imread(image_file, iread_mode)

    # 按读取模式收集结果
    if iread_mode == cv2.IMREAD_COLOR or iread_mode == cv2.IMREAD_ANYCOLOR:
        rows, cols, colors = img.shape
    else:
        rows, cols = img.shape

    # 计算旋转仿射矩阵
    matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 0.6)
    # 执行仿射变形
    img_rotate = cv2.warpAffine(img, matrix, (1 * cols, 1 * rows))
    show_image(img_rotate)


def test_image_flip():
    """
    翻转图像。
    使用函数cv2.flip(img,flipcode)翻转图像，flipcode控制翻转效果。
    flipcode = 0：沿x轴翻转
    flipcode > 0：沿y轴翻转
    flipcode < 0：x,y轴同时翻转
    :return:
    """
    image_file = os.path.join(image_dir, "demo1.png")
    img = cv2.imread(image_file)

    img_copy = img.copy()
    img_flip1 = cv2.flip(img, 0)
    img_flip2 = cv2.flip(img, 1)
    img_flip3 = cv2.flip(img, -1)

    images = [img_copy, img_flip1, img_flip2, img_flip3]
    pil_image_demo.plt_images(images)


def gamma_trans(img, gamma):
    """
    定义Gamma矫正的函数。
    :param img:
    :param gamma:
    :return:
    """
    # 具体做法是先归一化到1，然后gamma作为指数值求出新的像素值再还原
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)

    # 实现这个映射用的是OpenCV的查表函数
    return cv2.LUT(img, gamma_table)


def test_image_trans():
    """
    测试图像变换
    :return:
    """
    image_file = os.path.join(image_dir, "demo1.png")
    img = cv2.imread(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, dim = img.shape
    # 执行Gamma矫正，小于1的值让暗部细节大量提升，同时亮部细节少量提升
    img_gamma = gamma_trans(img, 0.5)

    # 沿着横纵轴放大1.6倍，然后平移(-150,-240)，最后沿原图大小截取，等效于裁剪并放大
    m_crop = np.array([
        [1.6, 0, -150],
        [0, 1.6, -240]
    ], dtype=np.float32)

    img_crop = cv2.warpAffine(img, m_crop, (width, height))

    # x轴的剪切变换，角度15°
    theta = 15 * np.pi / 180
    m_shear = np.array([
        [1, np.tan(theta), 0],
        [0, 1, 0]
    ], dtype=np.float32)

    img_sheared = cv2.warpAffine(img, m_shear, (width, height))

    # 顺时针旋转，角度15°
    m_rotate = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0]
    ], dtype=np.float32)

    img_rotated = cv2.warpAffine(img, m_rotate, (width, height))

    # show images
    images = [img, img_gamma, img_crop, img_sheared, img_rotated]
    pil_image_demo.plt_images(images)


def test_video_capture():
    """
    测试视频捕获。
    :return:
    """
    interval = 60  # 捕获图像的间隔，单位：秒
    num_frames = 100  # 捕获图像的总帧数
    out_fps = 24  # 输出文件的帧率

    # VideoCapture(0)表示打开默认的相机
    cap = cv2.VideoCapture(0)

    # 获取捕获的分辨率
    cap_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    file_path = os.path.join(com_config.RESOURCE_DIR, "my_video.avi")

    # 设置要保存视频的编码，分辨率和帧率
    video = cv2.VideoWriter(
        file_path,
        cv2.VideoWriter_fourcc('M', 'P', '4', '2'),
        out_fps,
        cap_size
    )

    # 对于一些低画质的摄像头，前面的帧可能不稳定，略过
    for i in range(42):
        cap.read()

    # 开始捕获，通过read()函数获取捕获的帧
    try:
        for i in range(num_frames):
            _, frame = cap.read()
            video.write(frame)

            # 如果希望把每一帧也存成文件，比如制作GIF，则取消下面的注释
            # filename = '{:0>6d}.png'.format(i)
            # cv2.imwrite(filename, frame)
            # 显示捕获到的图像
            # cv2.imshow('frame', frame)
            print('Frame {} is captured.'.format(i))
            # time.sleep(interval)
    except KeyboardInterrupt:
        # 提前停止捕获
        print('Stopped! {}/{} frames captured!'.format(i, num_frames))

    # 释放资源并写入视频文件
    video.release()
    cap.release()


def test_canny_edge():
    """
    canny边缘检测。
    :return:
    """
    image_file = os.path.join(image_dir, "demo1.png")
    img = cv2.imread(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 高斯模糊,降低噪声
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    # 灰度图像
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
    # 图像梯度
    x_grad = cv2.Sobel(gray, cv2.CV_16SC1, 1, 0)
    y_grad = cv2.Sobel(gray, cv2.CV_16SC1, 0, 1)
    # 计算边缘
    # 50和150参数必须符合1：3或者1：2
    edge_output = cv2.Canny(x_grad, y_grad, 50, 150)
    cv2.imshow("edge", edge_output)

    dst = cv2.bitwise_and(img, img, mask=edge_output)
    cv2.imshow('cedge', dst)

    images = [img, blurred, gray, edge_output, dst]
    pil_image_demo.plt_images(images)


if __name__ == "__main__":
    # show_image()
    # plot_images()
    # convert_image_color()
    # test_image_resize()
    # test_image_rotate()
    # test_image_flip()
    # test_image_cut_makeborder()
    # test_image_trans()
    # test_video_capture()
    test_canny_edge()
    pass

