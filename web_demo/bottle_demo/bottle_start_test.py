# -*- coding:utf-8 -*-

from bottle import Bottle, run, mako_view, request


myapp = Bottle()


@myapp.get('/hello/:name/:count#\\d+#')
@mako_view('hello')
def hello(name, count):
    ip = request.environ.get('REMOTE_ADDR')
    return dict(n=name, c=int(count), ip=ip)

run(app=myapp)