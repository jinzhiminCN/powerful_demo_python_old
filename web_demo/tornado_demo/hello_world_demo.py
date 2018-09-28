# -*- coding:utf-8 -*-

# ==============================================================================
# 测试tornado的基本用法。
# 1. tornado.web是tornado的基础web框架模块：
#     RequestHandler封装了对应一个请求的所有信息和方法，write(响应信息)就是写响应信息的一个方法。
#     Application是与服务器对接的接口，里面保存了路由信息表，其初始化接收的第一个参数就是一个路由信息映射元组的列表；
# 其listen(端口)方法用来创建一个http服务器实例，并绑定到给定端口（注意：此时服务器并未开启监听）。
# 2. tornado.ioloop是tornado的核心io循环模块，封装了Linux的epoll和BSD的kqueue，tornado高性能的基石。
#     IOLoop.current()返回当前线程的IOLoop实例。
#     IOLoop.start()启动IOLoop实例的I/O循环,同时服务器监听被打开。
# 3. tornado.httpserver是tornado的HTTP服务器实现。
# 4. tornado.options模块用来实现全局参数的定义、存储、转换。
#     options.define()用来定义options选项变量的方法，定义的变量可以在全局的tornado.options.options中获取使用。
#     options.options是全局的options对象，所有定义的选项变量都会作为该对象的属性。
#     options.parse_command_line()转换命令行参数，并将转换后的值对应的设置到全局options对象相关属性上。
#   追加命令行参数的方式是--myoption=myvalue。
# ==============================================================================
import tornado.ioloop
import tornado.web
import tornado.httpserver
from tornado.web import RequestHandler
from tornado.options import define, options, parse_command_line

# 定义服务器监听端口选项
define("port", default=8000, type=int, help="run server on the given port.")
# 无意义，演示多值情况
define("subject", default=[], type=str, multiple=True, help="subjects.")


class MainHandler(RequestHandler):
    def get(self):
        self.write("Hello, world")


class IndexHandler(RequestHandler):
    def get(self):
        python_url = self.reverse_url("python_url")
        self.write('<a href="{0}">subject</a>'.format(python_url))


class SubjectHandler(RequestHandler):
    def initialize(self, subject):
        self.subject = subject

    def get(self):
        self.write(self.subject)

handlers = [
    ("/", IndexHandler),
    ("/cpp", SubjectHandler, {"subject": "c++"}),
    tornado.web.url("/python", SubjectHandler, {"subject": "python"}, name="python_url")
]


def make_app():
    return tornado.web.Application([
        ("/", MainHandler),
    ])


def test_hello_world_app():
    """
    测试hello world的简单应用。
    :return:
    """
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()


def test_hello_world_server():
    """
    测试hello world的简单httpserver应用。
    :return:
    """
    app = make_app()
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(8888)
    tornado.ioloop.IOLoop.current().start()


def test_multi_process():
    """
    测试hello world应用的多进程服务。
    :return:
    """
    app = make_app()
    http_server = tornado.httpserver.HTTPServer(app)
    # 将服务器绑定到指定端口
    http_server.bind(8888)
    # 指定开启几个进程，默认值为1，即默认仅开启一个进程；
    # 如果为None或者<=0，则自动根据机器硬件的cpu核芯数创建同等数目的子进程；
    # 如果num_processes>0，则创建num_processes个子进程
    http_server.start(0)
    tornado.ioloop.IOLoop.current().start()


def test_options():
    """
    测试options选项在web应用中的用法。
    命令python hello_world_demo.py llo--port=9000 --subject=python,c++,java,php,ios
    :return:
    """
    parse_command_line()
    print("subject:{0}".format(options.subject))
    app = make_app()
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.current().start()


def test_multi_handler():
    """
    测试http服务中的多种handler方法。
    :return:
    """
    app = tornado.web.Application(handlers,
                                  debug=True)
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(8888)
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    pass
    # test_options()
    # test_hello_world_server()
    test_multi_handler()

