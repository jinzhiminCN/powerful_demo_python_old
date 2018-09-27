# -*- encoding:utf-8 -*-

# ==============================================================================
# 字符视频的处理。
# ==============================================================================
import os
import cv2
import subprocess
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
from PIL import Image, ImageFont, ImageDraw
from image_process.char_image import *

ascii_char = list("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:oa+>!:+. ")


def video_2_txt_jpg(file_name):
    """
    将视频拆分成图片。
    :param file_name:
    :return:
    """
    vc = cv2.VideoCapture(0)
    cnt = 1
    if vc.isOpened():
        ret_val, frame = vc.read()
        if not os.path.exists('Cache'):
            os.mkdir('Cache')
        os.chdir('Cache')
    else:
        ret_val = False

    while ret_val:
        cv2.imwrite(str(cnt) + '.jpg', frame)
        # 同时转换为ascii图
        generate_txt_image(str(cnt) + '.jpg')
        ret_val, frame = vc.read()
        cnt += 1

    os.chdir('..')

    return vc


def jpg_2_video(outfile_name, fps):
    """
    将图片合成视频。
    :param outfile_name:
    :param fps:
    :return:
    """
    fourcc = cv2.VideoWriter_fourcc(*"MP42")
    # cv2.VideoWriter_fourcc(*"MJPG")
    # cv2.VideoWriter_fourcc('M', 'P', '4', '2')

    images = os.listdir('Cache')
    im = Image.open('Cache/'+images[0])
    vw = cv2.VideoWriter(outfile_name + '.avi', fourcc, fps, im.size)

    os.chdir('Cache')
    for i_img in range(len(images)):
        image_name = "{0}_txt.jpg".format(str(i_img+1))
        frame = cv2.imread(image_name)
        vw.write(frame)
        common_logger.info("{0} finished".format(image_name))
    os.chdir('..')

    vw.release()


def video_2_mp3(file_name):
    """
    调用ffmpeg获取mp3音频文件。
    :param file_name:
    :return:
    """
    outfile_name = file_name.split('.')[0] + '.mp3'
    cmd = "ffmpeg -i {0} -f mp3 {1}".format(file_name, outfile_name)
    subprocess.call(cmd, shell=True)


def video_add_mp3(file_name, mp3_file):
    """
    合成音频和视频文件。
    :param file_name:
    :param mp3_file:
    :return:
    """
    outfile_name = file_name.split('.')[0] + '_txt.mp4'
    cmd = 'ffmpeg -i {0} -i {1} -strict -2 -f mp4 {2}'.format(file_name, mp3_file, outfile_name)
    subprocess.call(cmd, shell=True)


def test_char_video():
    """
    测试字符视频。
    :return:
    """
    vc = cv2.VideoCapture(0)
    # 获取帧率
    fps = 48

    dir_path = image_dir
    file_name = 'video.avi'
    file_path = os.path.join(dir_path, file_name)
    file_path_array = file_path.split('.')

    # 视频转换为字符图像
    # vc = video_2_txt_jpg(file_path)
    # vc.release()

    # 字符图像转换为视频
    jpg_2_video(file_path_array[0], fps)

    video_2_mp3(file_path)
    video_add_mp3(file_path_array[0]+'.avi', file_path_array[0]+'.mp3')

if __name__ == "__main__":
    test_char_video()
    pass
