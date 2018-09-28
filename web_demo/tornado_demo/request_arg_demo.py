# -*- coding:utf-8 -*-

# ==============================================================================
# 测试tornado.web.RequestHandler的基本用法。
# 利用HTTP协议向服务器传参有几种途径:
#   1).查询字符串(query string)，形如key1=value1&key2=value2；
#   2).请求体(body)中发送的数据，比如表单数据、json、xml；
#   3).提取uri的特定部分，如/blog/2016/09/0001，可以在服务器端的路由中用正则表达式截取；
#   4).在http报文的头(header)中增加自定义字段，如X-XSRFToken=value。
# 1. 获取查询字符串参数
#   get_query_argument(name, default=_ARG_DEFAULT, strip=True)
#   从请求的查询字符串中返回指定参数name的值，如果出现多个同名参数，则返回最后一个的值。
#   get_query_arguments(name, strip=True)
#   从请求的查询字符串中返回指定参数name的值，注意返回的是list列表。若未找到name参数，则返回空列表[]。
# 2. 获取请求体参数
#   get_body_argument(name, default=_ARG_DEFAULT, strip=True)
#   从请求体中返回指定参数name的值，如果出现多个同名参数，则返回最后一个的值。
#   get_body_arguments(name, strip=True)
#   对于请求体中的数据要求为字符串，且格式为表单编码格式（与url中的请求字符串格式相同），即key1=value1&key2=value2，
#   HTTP报文头Header中的"Content-Type"为application/x-www-form-urlencoded 或 multipart/form-data。
#   对于请求体数据为json或xml的，无法通过这两个方法获取。
# 3. 请求的其他信息
#   method 就是HTTP的请求方式，如GET或POST;
#   host 被请求的主机名;
#   uri 请求的完整资源标示，包括路径和查询字符串;
#   path 请求的路径部分;
#   query 请求的查询字符串部分;
#   version 使用的HTTP版本;
#   headers 请求的协议头，是类字典型的对象，支持关键字索引的方式获取特定协议头信息，
#     例如：request.headers["Content-Type"]body 请求体数据;
#   remote_ip 客户端的IP地址;
#   files 用户上传的文件，为字典类型，型如：
#     {
#       "form_filename1":[<tornado.httputil.HTTPFile>, <tornado.httputil.HTTPFile>],
#       "form_filename2":[<tornado.httputil.HTTPFile>,],
#     }
#     tornado.httputil.HTTPFile是接收到的文件对象，它有三个属性：
#     filename 文件的实际名字，与form_filename1不同，字典中的键名代表的是表单对应项的名字；
#     body 文件的数据实体；content_type 文件的类型。
#     这三个对象属性可以像字典一样支持关键字索引，如request.files["form_filename1"][0]["body"]。
# 4. 正则提取uri
#   对于路由映射支持正则提取uri，提取出来的参数会作为RequestHandler中对应请求方式的成员方法参数。
#   若在正则表达式中定义了名字，则参数按名传递；若未定义名字，则参数按顺序传递。
# ==============================================================================
import tornado.ioloop
import tornado.web
import tornado.httpserver
from tornado.web import RequestHandler, MissingArgumentError
from tornado.options import define, options, parse_command_line


class IndexHandler(RequestHandler):
    def get(self):
        # 获取get方式传递的参数
        user_name = self.get_query_argument("username")
        user_names = self.get_query_arguments("username")
        print(user_name)
        print(user_names)

        # 设置header信息
        self.set_header("Content-type", "application/json; charset=utf-8")
        self.set_cookie("loginuser", "admin")
        # self.send_error(404, msg="页面丢失", info="服务器异常")

    def post(self):
        body_arg = self.get_body_argument("body_a")
        body_args = self.get_body_arguments("body_a", strip=False)
        arg = self.get_argument("value_a")
        args = self.get_arguments("value_a")

        default_arg = self.get_argument("default_b", "some_value")
        default_args = self.get_arguments("default_b")

        try:
            missing_arg = self.get_argument("missing_c")
        except MissingArgumentError as e:
            missing_arg = "Catched the MissingArgumentError!"
            print(e)
        missing_args = self.get_arguments("missing_c")

        rep = "body_arg:{}<br/>".format(body_arg)
        rep += "body_args:{}<br/>".format(body_args)
        rep += "arg:{}<br/>".format(arg)
        rep += "args:{}<br/>".format(args)
        rep += "default_arg:{}<br/>".format(default_arg)
        rep += "default_args:{}<br/>".format(default_args)
        rep += "missing_arg:{}<br/>".format(missing_arg)
        rep += "missing_args:{}<br/>".format(missing_args)

        self.write(rep)


class UploadHandler(RequestHandler):
    def post(self):
        files = self.request.files
        img_files = files.get('img')
        if img_files:
            img_file = img_files[0]["body"]
            file = open("./img_file.png", 'wb+')
            file.write(img_file)
            file.close()
        self.write("OK")


class SubjectCityHandler(RequestHandler):
    def get(self, subject, city):
        self.write(("Subject: {0}<br/>City: {1}".format(subject, city)))


class SubjectDateHandler(RequestHandler):
    def get(self, date, subject):
        self.write(("Date: {0}<br/>Subject: {1}".format(date, subject)))


handlers = [
    ("/", IndexHandler),
    ("/upload", UploadHandler),
    ("/sub-city/(.+)/([a-z]+)", SubjectCityHandler),  # 无名方式
    ("/sub-date/(?P<subject>.+)/(?P<date>\d+)", SubjectDateHandler),  # 命名方式
]


def test_arg_server():
    """
    测试hello world的简单httpserver应用。
    :return:
    """
    app = tornado.web.Application(handlers)
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(8888)
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    pass
    test_arg_server()
