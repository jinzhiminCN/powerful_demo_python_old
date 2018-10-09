# -*- coding:utf-8 -*-

# ==============================================================================
# 测试smtp的相关方法。
# SMTP是发送邮件的协议，Python内置对SMTP的支持，可以发送纯文本邮件、HTML邮件以及带附件的邮件。
# Python对SMTP支持有smtplib和email两个模块，email负责构造邮件，smtplib负责发送邮件。
# ==============================================================================
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email import encoders
from email.header import Header
from email.utils import parseaddr, formataddr
import smtplib
import os
import time
import config.common_config as com_config

resource_dir = com_config.RESOURCE_DIR
image_dir = os.path.join(resource_dir, "image_data")


def send_email(subject):
    """
    发送邮件。
    :param subject: 邮件标题
    :return:
    """
    # 服务器地址
    email_host = ""
    # 发件人
    sender = ""
    # 密码，如果是授权码就填授权码
    password = ""
    # 收件人
    receiver = ""

    msg = MIMEMultipart()
    # 标题
    msg['Subject'] = subject
    # 发件人昵称
    msg['From'] = ""
    # 收件人昵称
    msg['To'] = ""

    signature = '''
        \n\t this is auto test report!
        \n\t you don't need to follow
    '''

    # 签名
    # text = MIMEText(signature, 'plain')
    # msg.attach(text)

    # 正文-图片 只能通过html格式来放图片
    mail_msg = '''
        <p>\n\t this is auto test report!</p>
        <p>\n\t you don't need to follow</p>
        <p>截图如下：</p>
        <p><img src="cid:image1"></p>
        '''
    msg.attach(MIMEText(mail_msg, 'html', 'utf-8'))

    img_path = os.path.join(image_dir, "demo1.png")
    # 指定图片为当前目录
    fp = open(img_path, 'rb')
    msg_image = MIMEImage(fp.read())
    fp.close()
    # 定义图片 ID，在 HTML 文本中引用
    msg_image.add_header('Content-ID', '<image1>')
    msg.attach(msg_image)

    ctype = 'application/octet-stream'
    maintype, subtype = ctype.split('/', 1)

    # 附件-图片
    image = MIMEImage(open(img_path, 'rb').read(), _subtype=subtype)
    image.add_header('Content-Disposition', 'attachment', filename='img.jpg')
    msg.attach(image)

    # 附件-文件
    file = MIMEBase(maintype, subtype)
    file.set_payload(open(r'320k.txt', 'rb').read())
    file.add_header('Content-Disposition', 'attachment', filename='test.txt')
    encoders.encode_base64(file)
    msg.attach(file)

    # 发送
    smtp = smtplib.SMTP()
    smtp.connect(email_host, 25)
    smtp.login(sender, password)
    smtp.sendmail(sender, receiver, msg.as_string())
    smtp.quit()
    print('success')


def create_msg():
    """
    创建email消息。
    :return:
    """
    msg = MIMEText('<html><body><h1>Hello</h1>' +
        '<p>send by <a href="http://www.python.org">Python</a>...</p>' +
        '</body></html>', 'html', 'utf-8')

    msg.attach(MIMEText('<html><body><h1>Hello</h1>' +
        '<p><img src="cid:0"></p>' +
        '</body></html>', 'html', 'utf-8'))

    # 利用MIMEMultipart就可以组合一个HTML和Plain，要注意指定subtype是alternative：
    msg = MIMEMultipart('alternative')
    msg['From'] = ...
    msg['To'] = ...
    msg['Subject'] = ...

    msg.attach(MIMEText('hello', 'plain', 'utf-8'))
    msg.attach(MIMEText('<html><body><h1>Hello</h1></body></html>', 'html', 'utf-8'))


def ssl_mail():
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    server = smtplib.SMTP(smtp_server, smtp_port)
    # 只需要在创建SMTP对象后，立刻调用starttls()方法，就创建了安全连接。
    server.starttls()
    # 剩下的代码和前面的一模一样:
    server.set_debuglevel(1)


def createMail():
    msg = MIMEText('hello, send by Python...', 'plain', 'utf-8')
    # 输入Email地址和口令:
    from_addr = input('From: ')
    password = input('Password: ')
    # 输入收件人地址:
    to_addr = input('To: ')
    # 输入SMTP服务器地址:
    smtp_server = input('SMTP server: ')

    # 这是因为邮件主题、如何显示发件人、收件人等信息并不是通过SMTP协议发给MTA，
    # 而是包含在发给MTA的文本中的，所以，必须把From、To和Subject添加到MIMEText中，才是一封完整的邮件：
    msg['From'] = _format_addr('Python爱好者 <%s>' % from_addr)
    msg['To'] = _format_addr('管理员 <%s>' % to_addr)
    msg['Subject'] = Header('来自SMTP的问候……', 'utf-8').encode()

    server = smtplib.SMTP(smtp_server, 25) # SMTP协议默认端口是25
    server.set_debuglevel(1)
    server.login(from_addr, password)
    server.sendmail(from_addr, [to_addr], msg.as_string())
    server.quit()


def _format_addr(s):
    name, addr = parseaddr(s)
    return formataddr((Header(name, 'utf-8').encode(), addr))


def createPartMail():
    # 输入Email地址和口令:
    from_addr = input('From: ')
    password = input('Password: ')
    # 输入收件人地址:
    to_addr = input('To: ')
    # 输入SMTP服务器地址:
    smtp_server = input('SMTP server: ')

    # 邮件对象:
    msg = MIMEMultipart()
    msg['From'] = _format_addr('Python爱好者 <%s>' % from_addr)
    msg['To'] = _format_addr('管理员 <%s>' % to_addr)
    msg['Subject'] = Header('来自SMTP的问候……', 'utf-8').encode()

    # 邮件正文是MIMEText:
    msg.attach(MIMEText('send with file...', 'plain', 'utf-8'))

    # 添加附件就是加上一个MIMEBase，从本地读取一个图片:
    with open('/Users/michael/Downloads/test.png', 'rb') as f:
        # 设置附件的MIME和文件名，这里是png类型:
        mime = MIMEBase('image', 'png', filename='test.png')
        # 加上必要的头信息:
        mime.add_header('Content-Disposition', 'attachment', filename='test.png')
        mime.add_header('Content-ID', '<0>')
        mime.add_header('X-Attachment-Id', '0')
        # 把附件的内容读进来:
        mime.set_payload(f.read())
        # 用Base64编码:
        encoders.encode_base64(mime)
        # 添加到MIMEMultipart:
        msg.attach(mime)


def test_send_email():
    """
    测试发送简单的email。
    :return:
    """
    now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    subject = now + '自动化测试报告'
    send_email(subject)




if __name__ == "__main__":
    pass

