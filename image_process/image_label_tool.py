# -*- encoding:utf-8 -*-

# ==============================================================================
# 图像标注工具。使用tkinter实现的。
# ==============================================================================
import os
import glob
import random
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

COLORS = ['red', 'blue', 'yellow', 'pink', 'cyan', 'green', 'black']
# image sizes for the examples
SIZE = 256, 256

# 指定缩放后的图像大小
SHOW_HEIGHT, SHOW_WIDTH = 800, 800


class LabelTool():
    def __init__(self, master):
        # set up the main frame
        self.parent = master
        self.parent.title("LabelTool")
        self.frame = Frame(self.parent)
        self.frame.pack(fill=BOTH, expand=1)
        self.parent.resizable(width=TRUE, height=TRUE)

        # initialize global state
        self.dir_name = ''
        self.image_dir = ''
        self.image_list = []

        self.bbox_dir = ''
        self.image_index = 0
        self.image_total = 0
        self.category = 0
        self.image_name = ''
        self.label_file_name = ''
        self.tk_img = None
        self.pil_image = None

        # initialize mouse state
        self.state = {}
        self.state['click'] = 0
        self.state['x'], self.state['y'] = 0, 0

        # reference to bbox
        self.bbox_id_list = []
        self.bbox_id = None
        self.bbox_list = []
        self.h_line = None
        self.v_line = None

        # ----------------- GUI stuff ---------------------
        # dir entry & load
        self.label = Label(self.frame, text="Image Dir:")
        self.label.grid(row=0, column=0, sticky=E)
        self.entry = Entry(self.frame)
        self.entry.grid(row=0, column=1, sticky=W + E)
        self.ldBtn = Button(self.frame, text="Load", command=self.load_dir)
        self.ldBtn.grid(row=0, column=2, sticky=W + E)

        # main panel for labeling
        self.mainPanel = Canvas(self.frame, cursor='tcross')
        self.mainPanel.bind("<Button-1>", self.mouse_click)
        self.mainPanel.bind("<Motion>", self.mouse_move)
        self.parent.bind("<Escape>", self.cancel_bbox)  # press <Escape> to cancel current bbox
        self.parent.bind("s", self.cancel_bbox)
        self.parent.bind("a", self.prev_image)  # press 'a' to go backward
        self.parent.bind("d", self.next_image)  # press 'd' to go forward
        self.mainPanel.grid(row=1, column=1, rowspan=4, sticky=W + N)

        # showing bbox info & delete bbox
        self.lb1 = Label(self.frame, text='Bounding boxes:')
        self.lb1.grid(row=1, column=2, sticky=W + N)

        self.listbox = Listbox(self.frame, width=28, height=12)
        self.listbox.grid(row=2, column=2, sticky=N)

        self.btnDel = Button(self.frame, text='Delete', command=self.del_bbox)
        self.btnDel.grid(row=3, column=2, sticky=W + E + N)
        self.btnClear = Button(self.frame, text='ClearAll', command=self.clear_bbox)
        self.btnClear.grid(row=4, column=2, sticky=W + E + N)

        # control panel for image navigation
        self.ctrPanel = Frame(self.frame)
        self.ctrPanel.grid(row=5, column=1, columnspan=2, sticky=W + E)
        self.prevBtn = Button(self.ctrPanel, text='<< Prev', width=10, command=self.prev_image)
        self.prevBtn.pack(side=LEFT, padx=5, pady=3)
        self.nextBtn = Button(self.ctrPanel, text='Next >>', width=10, command=self.next_image)
        self.nextBtn.pack(side=LEFT, padx=5, pady=3)
        self.progLabel = Label(self.ctrPanel, text="Progress:     /    ")
        self.progLabel.pack(side=LEFT, padx=5)
        self.tmpLabel = Label(self.ctrPanel, text="Go to Image No.")
        self.tmpLabel.pack(side=LEFT, padx=5)
        self.idxEntry = Entry(self.ctrPanel, width=5)
        self.idxEntry.pack(side=LEFT)
        self.goBtn = Button(self.ctrPanel, text='Go', command=self.goto_image)
        self.goBtn.pack(side=LEFT)

        # display mouse position
        self.disp = Label(self.ctrPanel, text='')
        self.disp.pack(side=RIGHT)

        self.frame.columnconfigure(1, weight=1)
        self.frame.rowconfigure(4, weight=1)

    def load_dir(self):
        """
        加载标注目录。目录下面应该包含images和bboxes目录。
        :return:
        """
        # 打开总目录
        self.dir_name = filedialog.askdirectory(title='打开目录')
        self.entry.insert(0, self.dir_name)

        # 图像目录
        self.image_dir = os.path.join(self.dir_name, 'images')

        self.image_list = glob.glob(os.path.join(self.image_dir, '*.jpg')) \
            + glob.glob(os.path.join(self.image_dir, '*.png'))
        if len(self.image_list) == 0:
            print('No .jpg images found in the specified dir!')
            return
        else:
            print('num={0}'.format(len(self.image_list)))

        self.image_index = 1
        self.image_total = len(self.image_list)

        # 标签目录
        self.bbox_dir = os.path.join(self.dir_name, 'bboxes')
        if not os.path.exists(self.bbox_dir):
            os.mkdir(self.bbox_dir)

        # 加载图像
        self.load_image()
        print('{0} images loaded from {1}'.format(self.image_total, self.image_dir))

    def load_image(self):
        """
        加载图像。
        :return:
        """
        # load image
        image_path = self.image_list[self.image_index - 1]
        self.pil_image = Image.open(image_path)

        # 获取图像的原始大小
        w0, h0 = self.pil_image.size
        print("图像的宽度:{0}, 高度:{1}".format(w0, h0))

        # 等比例缩放到指定大小
        if w0 > SHOW_WIDTH or h0 > SHOW_HEIGHT:
            self.pil_image.thumbnail((SHOW_WIDTH, SHOW_HEIGHT), Image.ANTIALIAS)

        self.tk_img = ImageTk.PhotoImage(self.pil_image)

        self.mainPanel.config(width=max(self.tk_img.width(), 400), height=max(self.tk_img.height(), 400))
        self.mainPanel.create_image(0, 0, image=self.tk_img, anchor=NW)
        self.progLabel.config(text="{0:0>4}{1:0>4}".format(self.image_index, self.image_total))

        # load labels
        self.clear_bbox()
        self.image_name = os.path.split(image_path)[-1].split('.')[0]
        label_name = self.image_name + '.txt'
        self.label_file_name = os.path.join(self.bbox_dir, label_name)
        if os.path.exists(self.label_file_name):
            with open(self.label_file_name, 'r', encoding='utf-8') as f:
                for (i, line) in enumerate(f):
                    print(line)
                    tmp = [(t.strip()) for t in line.split()]

                    if len(tmp) >= 4 :
                        # tmp = (0.1, 0.3, 0.5, 0.5)
                        print("tmp[0,1,2,3]==={0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}".format(
                              float(tmp[0]), float(tmp[1]), float(tmp[2]), float(tmp[3])))

                        self.bbox_list.append(tuple(tmp))
                        tmp[0] = float(tmp[0])
                        tmp[1] = float(tmp[1])
                        tmp[2] = float(tmp[2])
                        tmp[3] = float(tmp[3])

                        tx0 = int(tmp[0] * SHOW_WIDTH)
                        ty0 = int(tmp[1] * SHOW_HEIGHT)
                        tx1 = int(tmp[2] * SHOW_WIDTH)
                        ty1 = int(tmp[3] * SHOW_HEIGHT)
                        print("tx0={0}, ty0={1}, tx1={2}, ty1={3}".format(tx0, ty0, tx1, ty1))

                        tmp_id = self.mainPanel.create_rectangle(tx0, ty0, tx1, ty1, width=2,
                                                                 outline=COLORS[(len(self.bbox_list) - 1) % len(COLORS)])

                        self.bbox_id_list.append(tmp_id)
                        self.listbox.insert(END, '({0:.2f},{1:.2f})-({2:.2f},{3:.2f})'
                                            .format(tmp[0], tmp[1], tmp[2], tmp[3]))

                        self.listbox.itemconfig(len(self.bbox_id_list) - 1,
                                                fg=COLORS[(len(self.bbox_id_list) - 1) % len(COLORS)])

    def cancel_bbox(self, event):
        """
        取消bbox。
        :param event:
        :return:
        """
        if 1 == self.state['click']:
            if self.bbox_id:
                self.mainPanel.delete(self.bbox_id)
                self.bbox_id = None
                self.state['click'] = 0

    def save_image(self):
        """
        保存图像。
        :return:
        """
        print("bbox list:{0}".format(self.bbox_list))
        print(self.label_file_name)
        with open(self.label_file_name, 'w') as f:
            for bbox in self.bbox_list:
                f.write(' '.join(map(str, bbox)) + '\n')
        print('Image No.{0} saved'.format(self.image_index))

    def mouse_click(self, event):
        """
        鼠标点击操作。
        :param event:
        :return:
        """
        if self.state['click'] == 0:
            self.state['x'], self.state['y'] = event.x, event.y
        else:
            x1, x2 = min(self.state['x'], event.x), max(self.state['x'], event.x)
            y1, y2 = min(self.state['y'], event.y), max(self.state['y'], event.y)

            x1, x2 = x1 / SHOW_WIDTH, x2 / SHOW_WIDTH
            y1, y2 = y1 / SHOW_HEIGHT, y2 / SHOW_HEIGHT

            self.bbox_list.append((x1, y1, x2, y2))
            self.bbox_id_list.append(self.bbox_id)
            self.bbox_id = None
            self.listbox.insert(END, '({0:.2f}, {1:.2f})-({2:.2f}, {3:.2f})'.format(x1, y1, x2, y2))
            self.listbox.itemconfig(len(self.bbox_id_list) - 1, fg=COLORS[(len(self.bbox_id_list) - 1) % len(COLORS)])
        self.state['click'] = 1 - self.state['click']

    def mouse_move(self, event):
        """
        鼠标移动操作。
        :param event:
        :return:
        """
        self.disp.config(text='x: {0:.2f}, y: {1:.2f}'.format(event.x / SHOW_WIDTH, event.y / SHOW_HEIGHT))
        if self.tk_img:
            if self.h_line:
                self.mainPanel.delete(self.h_line)
            self.h_line = self.mainPanel.create_line(0, event.y, self.tk_img.width(), event.y, width=2)
            if self.v_line:
                self.mainPanel.delete(self.v_line)
            self.v_line = self.mainPanel.create_line(event.x, 0, event.x, self.tk_img.height(), width=2)
        if 1 == self.state['click']:
            if self.bbox_id:
                self.mainPanel.delete(self.bbox_id)
            self.bbox_id = self.mainPanel.create_rectangle(self.state['x'], self.state['y'], \
                                                           event.x, event.y, \
                                                           width=2, \
                                                           outline=COLORS[len(self.bbox_list) % len(COLORS)])

    def del_bbox(self):
        """
        删除bbox。
        :return:
        """
        sel = self.listbox.curselection()
        if len(sel) != 1:
            return
        idx = int(sel[0])
        print("delete bbox:{0}".format(idx))
        self.mainPanel.delete(self.bbox_id_list[idx])
        self.bbox_id_list.pop(idx)
        self.bbox_list.pop(idx)
        self.listbox.delete(idx)

    def clear_bbox(self):
        """
        清空bbox。
        :return:
        """
        for idx in range(len(self.bbox_id_list)):
            self.mainPanel.delete(self.bbox_id_list[idx])
        self.listbox.delete(0, len(self.bbox_list))
        self.bbox_id_list = []
        self.bbox_list = []

    def prev_image(self, event=None):
        """
        上一张图片。
        :param event:
        :return:
        """
        self.save_image()
        if self.image_index > 1:
            self.image_index -= 1
            self.load_image()

    def next_image(self, event=None):
        """
        下一张图片。
        :param event:
        :return:
        """
        self.save_image()
        if self.image_index < self.image_total:
            self.image_index += 1
            self.load_image()

    def goto_image(self):
        """
        跳转到图片。
        :return:
        """
        idx = int(self.idxEntry.get())
        if 1 <= idx <= self.image_total:
            self.save_image()
            self.image_index = idx
            self.load_image()


if __name__ == '__main__':
    root = Tk()
    tool = LabelTool(root)
    root.mainloop()
