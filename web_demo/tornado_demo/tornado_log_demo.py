# -*- coding:utf-8 -*-

# ==============================================================================
# 测试tornado的log基本用法。使用日志记录http访问记录。
# 1. 可以在Application中自定义一个log_function，修改http访问日志。
# 2. 重写RequestHandler的_request_summary方法。
# 3. 重写Application的log_request方法。
# ==============================================================================
import os
import logging
from logging.handlers import TimedRotatingFileHandler
import tornado.ioloop
import tornado.web
import tornado.httpserver
from tornado.web import access_log, RequestHandler
from tornado.options import define, options, parse_command_line


define("port", default=8800, help="run on the given port", type=int)
define("debug", default=False, help="run in debug mode")
define("env", default='dev', help="run env: dev, test, product")

service_abbr = ''


class MainHandler(RequestHandler):
    def get(self):
        self.write("Hello, world")

    def _request_summary(self):
        return "{0} {1} ({2}) {3}\n{4}"\
            .format(self.request.method, self.request.uri,
                    self.request.remote_ip, self.request.headers["User-Agent"],
                    self.request.arguments)

handlers = [
    ("/", MainHandler),
]


def log_func(handler):
    """
    外部日志处理函数。
    :param handler:
    :return:
    """
    if handler.get_status() < 400:
        log_method = access_log.info
    elif handler.get_status() < 500:
        log_method = access_log.warning
    else:
        log_method = access_log.error
    request_time = 1000.0 * handler.request.request_time()
    log_method("%d %s %s (%s) %s %s %.2fms",
               handler.get_status(), handler.request.method,
               handler.request.uri, handler.request.remote_ip,
               handler.request.headers["User-Agent"],
               handler.request.arguments,
               request_time)


class Application(tornado.web.Application):
    def __init__(self, **settings):
        tornado.web.Application.__init__(
            self, handlers,
            static_path=os.path.join(os.path.dirname(__file__), "static"),
            xsrf_cookies=True,
            gzip=True,
            debug=options.debug,
            env_name=options.env,
            service_abbr=service_abbr,
            autoreload=False,
            **settings)
        # 修改日志记录器
        self.init_logging("app.log")
        # 使用log_function参数
        settings = dict()
        settings["log_function"] = log_func

    @staticmethod
    def init_logging(log_file):
        """
        初始化通用日志输出。
        :param self:
        :param log_file:
        :return:
        """
        # 使用TimedRotatingFileHandler处理器
        file_handler = TimedRotatingFileHandler(log_file, when="d", interval=1, backupCount=30)
        # 输出格式
        log_formatter = logging.Formatter(
            "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] [%(lineno)d]  %(message)s"
        )
        file_handler.setFormatter(log_formatter)
        # 将处理器附加到根logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)

    def log_request(self, handler):
        """Writes a completed HTTP request to the logs.

        By default writes to the python root logger.  To change
        this behavior either subclass Application and override this method,
        or pass a function in the application settings dictionary as
        ``log_function``.
        """
        if options.debug:
            super(Application, self).log_request(handler)
            return
        if "log_function" in self.settings:
            self.settings["log_function"](handler)
            return
        if handler.get_status() < 400:
            log_method = tornado.web.access_log.info
        elif handler.get_status() < 500:
            log_method = tornado.web.access_log.warning
        else:
            log_method = tornado.web.access_log.error
        request_time = 1000.0 * handler.request.request_time()
        log_method("%s^|%s^|%s^|%s^|%s",
                   handler.request.remote_ip,
                   'Y' if handler.get_status() < 400 else 'N',
                   handler.get_status(),
                   request_time,
                   handler.request.method + ' ' + handler.request.uri)


def start():
    """
    启动项目。
    :return:
    """
    parse_command_line()
    app = Application()
    logging.info('127.0.0.1^|Y^|200^|0.1^|CUSTOM === Start ===')
    http_server = tornado.httpserver.HTTPServer(app, xheaders=True)
    http_server.bind(options.port)
    http_server.start(1)
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    start()
    pass
