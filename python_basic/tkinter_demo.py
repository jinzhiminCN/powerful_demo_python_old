# -*- coding:utf-8 -*-

# ==============================================================================
# 测试tkinter模块的相关方法。
# ==============================================================================
import time
import tkinter as tk
from tkinter import *
from datetime import date, timedelta, timezone
import datetime

from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()
# =========================== 全局常量 ===========================
root = tk.Tk()


class Application(tk.Frame):
    """
    在控制台打印hi的简单应用。
    """
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.hi_there = tk.Button(self)
        self.hi_there["text"] = "Hello World\n(click me)"
        self.hi_there["command"] = self.say_hi
        self.hi_there.pack(side="top")

        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=root.destroy)
        self.quit.pack(side="bottom")

    def say_hi(self):
        """
        打印hi。
        :return:
        """
        print("hi there, everyone!")


def frame(master, side):
    """
    创建横条型框架。
    :param master:
    :param side:
    :return:
    """
    frm = Frame(master)
    frm.pack(side=side, expand=YES, fill=BOTH)
    return frm


def button(master, side, text, command=None):
    """
    创建按钮。
    :param master:
    :param side:
    :param text:
    :param command:
    :return:
    """
    btn = Button(master, text=text, command=command)
    btn.pack(side=side, expand=YES, fill=BOTH)
    return btn


class Calculator(Frame):
    def __init__(self):
        Frame.__init__(self)

        self.pack(expand=YES, fill=BOTH)
        self.master.title('Simple Calculator')

        display = StringVar()
        # 添加输入框
        Entry(self, relief=SUNKEN, textvariable=display)\
            .pack(side=TOP, expand=YES, fill=BOTH)

        # 添加横条型框架以及里面的按钮
        for key in ('123', '456', '789', '-0.'):
            key_frm = frame(self, TOP)
            for char in key:
                button(key_frm, LEFT, char, lambda w=display, c=char: w.set(w.get() + c))

        # 添加操作符按钮
        ops_frm = frame(self, TOP)
        for char in '+-*/=':
            if char == '=':
                btn = button(ops_frm, LEFT, char)
                btn.bind('<ButtonRelease - 1>', lambda e, s=self, w=display: s.calc(w), '+')
            else:
                btn = button(ops_frm, LEFT, char, lambda w=display, s='{0}'.format(char): w.set(w.get() + s))

        # 添加清除按钮
        clear_frm = frame(self, BOTTOM)
        button(clear_frm, LEFT, 'clear', lambda w=display: w.set(''))

    def calc(self, display):
        """
        调用eval函数计算表达式的值。
        :param display:
        :return:
        """
        try:
            display.set(eval(display.get()))
        except:
            display.set("ERROR")


def test_application():
    """
    测试应用Application。
    :return:
    """
    app = Application(master=root)
    app.mainloop()


def test_calculator():
    """
    测试应用Calculator。
    :return:
    """
    Calculator().mainloop()


if __name__ == "__main__":
    pass
    # test_application()
    test_calculator()

