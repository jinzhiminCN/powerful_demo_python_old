# -*- coding:utf-8 -*-

# ==============================================================================
# 测试bottle的基本用法。
# 使用http://localhost:8080/hello/xxx来测试
# ==============================================================================

from bottle import route, run, template


@route('/hello/<name>')
def index(name):
    """
    返回hello请求的网页模板。
    :param name:
    :return:
    """
    return template('<b>Hello {{name}}</b>!', name=name)


def startup():
    """
    启动网站。
    :return:
    """
    run(host='localhost', port=8080)


if __name__ == "__main__":
    startup()
    pass
