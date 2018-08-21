# -*- coding:utf-8 -*-

# ==============================================================================
# 测试bottle的基本用法，使用模板。
# 使用http://localhost:8080/hello/xxx/1来测试
# ==============================================================================

from bottle import Bottle, run, mako_view, request


myapp = Bottle()


@myapp.get('/hello/:name/:count#\\d+#')
@mako_view('hello')
def hello(name, count):
    ip = request.environ.get('REMOTE_ADDR')
    return dict(n=name, c=int(count), ip=ip)

run(app=myapp)