# -*- encoding:utf-8 -*-

# ==============================================================================
# 图像标注模块。使用openCV模块实现标注。
# ==============================================================================
import os
import cv2
from PIL import Image


# 定义标注窗口的默认名称
WINDOW_NAME = 'Simple Bounding Box Labeling Tool'

# 定义画面刷新的大概帧率（是否能达到取决于电脑性能）
FPS = 24

# 定义支持的图像格式
SUPPORTED_FORMATS = ['jpg', 'jpeg', 'png']

# 定义默认物体框的名字为Object，颜色蓝色，当没有用户自定义物体时用默认物体
DEFAULT_COLOR = {'Object': (255, 0, 0)}

# 定义灰色，用于信息显示的背景和未定义物体框的显示
COLOR_GRAY = (192, 192, 192)
COLOR_BLUE = (0, 0, 255)

# 在图像下方多出BAR_HEIGHT这么多像素的区域用于显示文件名和当前标注物体等信息
BAR_HEIGHT = 16

# 上下左右，ESC及删除键对应的cv.waitKey()的返回值
# 注意这个值根据操作系统不同有不同，
KEY_UP = 65362
KEY_DOWN = 65364
KEY_LEFT = 65361
KEY_RIGHT = 65363
KEY_ESC = 27
KEY_DELETE = 65535

KEY_D = 100
KEY_H = 104
KEY_J = 106
KEY_K = 107
KEY_L = 108

# 空键用于默认循环
KEY_EMPTY = 0

LINE_THICKNESS = 1
get_bbox_name = '{}.bbox'.format


class SimpleBBoxLabeling:
    def __init__(self, data_dir, fps=FPS, window_name=None):
        """
        初始化。
        :param data_dir:
        :param fps:
        :param window_name:
        """
        self._data_dir = data_dir
        self.fps = fps
        self.window_name = window_name if window_name else WINDOW_NAME

        # pt0是正在画的左上角坐标，pt1是鼠标所在坐标
        self._pt0 = None
        self._pt1 = None

        # 表明当前是否正在画框的状态标记
        self._drawing = False

        # 当前标注物体的名称
        self._cur_label = None
        # 当前标注的图像
        self._img = None
        self._img_width = None
        self._img_height = None
        # 当前图像对应的所有已标注框
        self._bboxes = []
        # 选中的标注框
        self._cur_box = None

        # 如果有用户自定义的标注信息则读取，否则用默认的物体和颜色
        label_path = '{}.labels'.format(self._data_dir)
        self.label_colors = DEFAULT_COLOR if not os.path.exists(label_path) else self.load_labels(label_path)

        # 获取已经标注的文件列表和还未标注的文件列表
        image_files = [x for x in os.listdir(self._data_dir) if x[x.rfind('.') + 1:].lower() in SUPPORTED_FORMATS]
        labeled = [x for x in image_files if os.path.exists(get_bbox_name(os.path.join(self._data_dir, x)))]
        to_be_labeled = [x for x in image_files if x not in labeled]

        # 每次打开一个文件夹，都自动从还未标注的第一张开始
        self._file_list = labeled + to_be_labeled
        self._image_index = len(labeled)
        if self._image_index > len(self._file_list) - 1:
            self._image_index = len(self._file_list) - 1

    def _mouse_ops(self, event, x, y, flags, param):
        """
        鼠标回调函数。
        :param event: 鼠标事件
        :param x: 鼠标所在位置的x
        :param y: 鼠标所在位置的y
        :param flags:
        :param param:
        :return:
        """
        # 按下左键时，坐标为左上角，同时表明开始画框，改变drawing标记为True
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._pt0 = (x, y)

        # 实时更新右下角坐标方便画框
        elif event == cv2.EVENT_MOUSEMOVE:
            self._pt1 = (x, y)
            if self._drawing and flags == cv2.EVENT_FLAG_LBUTTON:
                # print("鼠标移动", event, flags)
                pass

        # 左键抬起，表明当前框画完了，坐标记为右下角，并保存，同时改变drawing标记为False
        elif event == cv2.EVENT_LBUTTONUP:
            # print("左键抬起", event, flags)
            self._drawing = False
            self._pt1 = (x, y)
            if self._pt0 and self._pt1:
                left, top = self._pt0
                right, bottom = self._pt1
                if left < right and top < bottom:
                    self._bboxes.append((self._cur_label, (self._pt0, self._pt1)))

        # 双击鼠标左键，选中一个标注框
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            # print("双击左键", event, flags)
            self._cur_box = None
            if self._bboxes:
                for i_box, _box in enumerate(self._bboxes):
                    pt0, pt1 = _box[1]
                    if pt0[0] <= x <= pt1[0] and pt0[1] <= y <= pt1[1]:
                        self._cur_box = i_box
                        break
            print("选中的标注框：{0}".format(self._cur_box))

        # 鼠标右键删除最近选中的框
        elif event == cv2.EVENT_RBUTTONUP:
            if self._bboxes and self._cur_box is not None \
                    and 0 <= self._cur_box < len(self._bboxes):
                self._bboxes.pop(self._cur_box)
                self._cur_box = None
            elif self._bboxes and self._cur_box is None:
                self._bboxes.pop()

    def _clean_bbox(self):
        """
        清除所有标注框和当前状态。
        :return:
        """
        self._pt0 = None
        self._pt1 = None
        self._drawing = False
        self._bboxes = []

    def _draw_bbox(self, img):
        """
        画标注框和当前信息的函数。
        :param img:
        :return:
        """
        # 在图像下方多出BAR_HEIGHT这么多像素的区域用于显示文件名和当前标注物体等信息
        h, w = img.shape[:2]
        canvas = cv2.copyMakeBorder(img, 0, BAR_HEIGHT, 0, 0, cv2.BORDER_CONSTANT, value=COLOR_GRAY)

        # 正在标注的物体信息，如果鼠标左键已经按下，则显示两个点坐标，否则显示当前待标注物体的名称
        label_msg = '{}: {}, {}'.format(self._cur_label, self._pt0, self._pt1) \
            if self._drawing \
            else 'Current label: {}'.format(self._cur_label)

        # 显示当前文件名，文件个数信息
        msg = '{}/{}: {} | {}'.format(self._image_index + 1, len(self._file_list),
                                      self._file_list[self._image_index], label_msg)
        cv2.putText(canvas, msg, (1, h + 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1)

        # 画出已经标好的框和对应名字
        for i_box, (label, (bpt0, bpt1)) in enumerate(self._bboxes):
            label_color = self.label_colors[label] if label in self.label_colors else COLOR_GRAY
            if i_box == self._cur_box:
                label_color = COLOR_BLUE
            cv2.rectangle(canvas, bpt0, bpt1, label_color, thickness=LINE_THICKNESS)
            cv2.putText(canvas, label, (bpt0[0] + 3, bpt0[1] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, label_color, LINE_THICKNESS)

        # 画正在标注的框和对应名字
        if self._drawing:
            label_color = self.label_colors[self._cur_label] if self._cur_label in self.label_colors else COLOR_GRAY
            if self._pt1[0] >= self._pt0[0] and self._pt1[1] >= self._pt0[1]:
                cv2.rectangle(canvas, self._pt0, self._pt1, label_color, thickness=LINE_THICKNESS)
            cv2.putText(canvas, self._cur_label, (self._pt0[0] + 3, self._pt0[1] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, label_color, LINE_THICKNESS)
        return canvas

    @staticmethod
    def load_labels(file_path):
        """
        加载物体及对应颜色信息的数据。
        :param file_path:
        :return:
        """
        label_colors = {}
        with open(file_path, 'r') as f:
            line = f.readline().rstrip()
            while line:
                label, color = eval(line)
                label_colors[label] = color
                line = f.readline().rstrip()
        return label_colors

    @staticmethod
    def load_bbox(file_path):
        """
        利用eval()读取标注框字符串到数据。
        :param file_path:
        :return:
        """
        bboxes = []
        with open(file_path, 'r') as f:
            line = f.readline().rstrip()
            while line:
                bboxes.append(eval(line))
                line = f.readline().rstrip()
        return bboxes

    @staticmethod
    def export_bbox(file_path, bboxes):
        """
        利用repr()导出标注框数据到文件。
        :param file_path:
        :param bboxes:
        :return:
        """
        if bboxes:
            with open(file_path, 'w') as f:
                for bbox in bboxes:
                    line = repr(bbox) + '\n'
                    f.write(line)
        elif os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def load_sample(file_path):
        """
        读取图像文件和对应标注框信息（如果有的话）。
        :param file_path:
        :return:
        """
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        bbox_file_path = get_bbox_name(file_path)
        bboxes = []
        if os.path.exists(bbox_file_path):
            bboxes = SimpleBBoxLabeling.load_bbox(bbox_file_path)
        return img, bboxes

    def _export_n_clean_bbox(self):
        """
        导出当前标注框信息并清空。
        :return:
        """
        bbox_file_path = os.sep.join([self._data_dir, get_bbox_name(self._file_list[self._image_index])])
        self.export_bbox(bbox_file_path, self._bboxes)
        self._clean_bbox()

    def _delete_current_sample(self):
        """
        删除当前标注框信息
        :return:
        """
        filename = self._file_list[self._image_index]
        file_path = os.sep.join([self._data_dir, filename])
        file_path = get_bbox_name(file_path)
        if os.path.exists(file_path):
            os.remove(file_path)
        self._file_list.pop(self._image_index)
        print('{} is deleted!'.format(filename))

    def _adjust_box_position(self, direction='left'):
        """
        调整标注框的位置。
        :param direction: 移动方向
        :return:
        """
        label_name, box_pt = self._bboxes[self._cur_box]
        pt0, pt1 = box_pt

        if direction == 'up':
            if pt0[1] > 1:
                pt0, pt1 = (pt0[0], pt0[1] - 1), (pt1[0], pt1[1] - 1)
                self._bboxes[self._cur_box] = (label_name, (pt0, pt1))
        elif direction == "down":
            if pt1[1] < self._img_height:
                pt0, pt1 = (pt0[0], pt0[1] + 1), (pt1[0], pt1[1] + 1)
                self._bboxes[self._cur_box] = (label_name, (pt0, pt1))
        elif direction == "left":
            if pt0[0] > 1:
                pt0, pt1 = (pt0[0] - 1, pt0[1]), (pt1[0] - 1, pt1[1])
                self._bboxes[self._cur_box] = (label_name, (pt0, pt1))
        elif direction == "right":
            if pt1[0] < self._img_height:
                pt0, pt1 = (pt0[0] + 1, pt0[1]), (pt1[0] + 1, pt1[1])
                self._bboxes[self._cur_box] = (label_name, (pt0, pt1))

    def start(self):
        """
        开始OpenCV窗口循环的方法，定义程序的主逻辑。
        :return:
        """
        # 之前标注的文件名，用于程序判断是否需要执行一次图像读取
        last_filename = ''
        # 当前标注的文件名
        filename = ''

        # 标注物体在列表中的下标
        label_index = 0

        # 所有标注物体名称的列表
        labels = self.label_colors.keys()
        labels = list(labels)

        # 待标注物体的种类数
        n_labels = len(labels)

        # 定义窗口和鼠标回调
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_ops)
        key = KEY_EMPTY

        # 定义每次循环的持续时间
        delay = int(1000 / FPS)

        # 只要没有按下Esc键，就持续循环
        while key != KEY_ESC:
            # 上下键(K, J)用于选择当前标注物体，或者调整标注框高度
            if key == KEY_K:
                # 使用按键调整位置
                if self._cur_box is not None:
                    self._adjust_box_position("up")
                else:
                    if label_index == 0:
                        pass
                    else:
                        label_index -= 1
                        label_index = max(label_index, 0)

            elif key == KEY_J:
                # 使用按键调整位置
                if self._cur_box is not None:
                    self._adjust_box_position("down")
                else:
                    if label_index == n_labels - 1:
                        pass
                    else:
                        label_index += 1
                        label_index = min(label_index, len(labels))

            # 左右键(H, L)切换当前标注的图片
            elif key == KEY_H:
                # 使用按键调整位置
                if self._cur_box is not None:
                    self._adjust_box_position("left")
                else:
                    # 已经到了第一张图片的话就不需要清空上一张
                    if self._image_index > 0:
                        self._export_n_clean_bbox()

                    self._image_index -= 1
                    if self._image_index < 0:
                        self._image_index = 0

            elif key == KEY_L:
                # 使用按键调整位置
                if self._cur_box is not None:
                    self._adjust_box_position("right")
                else:
                    # 已经到了最后一张图片的话就不需要清空上一张
                    if self._image_index < len(self._file_list) - 1:
                        self._export_n_clean_bbox()

                    self._image_index += 1
                    if self._image_index > len(self._file_list) - 1:
                        self._image_index = len(self._file_list) - 1

            # 删除当前图片和对应标注信息
            elif key == KEY_DELETE:
                self._delete_current_sample()
                key = KEY_EMPTY
                continue

            # 如果键盘操作执行了换图片，则重新读取，更新图片
            filename = self._file_list[self._image_index]
            # print("filename:{0}, last_filename:{1}".format(filename, last_filename))
            if filename != last_filename:
                file_path = os.sep.join([self._data_dir, filename])
                self._img, self._bboxes = self.load_sample(file_path)
                self._img_width, self._img_height = self._img.shape[:2]

            # 更新当前标注物体名称
            self._cur_label = labels[label_index]

            # 把标注和相关信息画在图片上并显示指定的时间
            canvas = self._draw_bbox(self._img)
            cv2.imshow(self.window_name, canvas)
            key = cv2.waitKey(delay)

            # 显示按键的id
            # if key != -1:
            #     print("当前按键:{0}".format(key))

            # 当前文件名就是下次循环的老文件名
            last_filename = filename

        print('Finished!')

        cv2.destroyAllWindows()
        print("导出BBoxes:", self._bboxes)
        # 如果退出程序，需要对当前进行保存
        export_box_path = os.sep.join([self._data_dir, get_bbox_name(filename)])
        self.export_bbox(export_box_path, self._bboxes)

        print('Labels updated!')


def crop_image_by_bbox(image_path, bbox_path, photo_result_dir):
    """
    根据bbox文件切分图像。
    :param image_path: 原始图像路径
    :param bbox_path: bbox路径
    :param photo_result_dir: 切分图像目录
    :return:
    """
    image_name = os.path.basename(image_path)
    image_src = Image.open(image_path)
    image_src = image_src.convert("L")

    index = 0
    with open(bbox_path, mode='r', encoding='utf-8') as label_file:
        for line in label_file:
            # 获取图像box
            line = line.replace(" ", "").replace("(", "").replace(")", "")
            line_split = line.split(",")
            box = [int(x) for x in line_split[1:]]
            index += 1

            # 将标注方框不符合要求的过滤掉
            left, top, right, bottom = box
            if left >= right or top >= bottom:
                continue

            # 剪切图像
            small_image = image_src.crop(box)

            # 保存图像
            small_image_name = "{0}_{1}.jpg".format(image_name[:-4], index)
            small_image_path = os.path.join(photo_result_dir, small_image_name)
            small_image.save(small_image_path)


def image_crop(image_src_dir, image_dst_dir):
    """
    切分图像。
    :param image_src_dir
    :param image_dst_dir
    :return:
    """
    for filename in os.listdir(image_src_dir):
        if filename[-4:] == 'bbox':
            continue
        else:
            image_path = os.path.join(image_src_dir, filename)
            bbox_path = "{0}.bbox".format(image_path)
            if os.path.exists(bbox_path):
                crop_image_by_bbox(image_path, bbox_path, image_dst_dir)


if __name__ == '__main__':
    image_dir = 'E:/images'
    labeling_task = SimpleBBoxLabeling(image_dir)
    labeling_task.start()
