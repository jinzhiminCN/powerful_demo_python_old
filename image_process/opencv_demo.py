# -*- coding:utf-8 -*-

# ==============================================================================
# 测试OpenCV cv2模块的相关方法。
# ==============================================================================
import cv2
import numpy as np
import os
import config.common_config as com_config


resource_dir = com_config.RESOURCE_DIR
image_dir = os.path.join(resource_dir, "image_data")

# =========================== test function ===========================


def show_image():
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

    win_name = 'example'
    cv2.namedWindow(win_name)
    cv2.imshow(win_name, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows(win_name)


if __name__ == "__main__":
    # show_image()
    plot_images()
    pass