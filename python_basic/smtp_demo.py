# -*- coding:utf-8 -*-

# ==============================================================================
# 测试smtp的相关方法。
# https://docs.python.org/3/library/email.examples.html
# SMTP(Simple Mail Transfer Protocol) MIME(Multipurpose Internet Mail Extensions)
# MTA(Mail Transfer Agent 邮件传输代理)
# SMTP是发送邮件的协议，Python内置对SMTP的支持，可以发送纯文本邮件、HTML邮件以及带附件的邮件。
# Python对SMTP支持有smtplib和email两个模块，email负责构造邮件，smtplib负责发送邮件。
# ==============================================================================
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.message import EmailMessage
from email import encoders
from email.header import Header
from email.utils import parseaddr, formataddr, make_msgid
import smtplib
import imghdr
import os
import time
import config.common_config as com_config
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()
resource_dir = com_config.RESOURCE_DIR
image_dir = os.path.join(resource_dir, "image_data")

# 服务器地址
email_host = com_config.SMTP_HOST
# 发件人
sender = com_config.SMTP_SENDER
# 密码，如果是授权码就填授权码
password = com_config.SMTP_PASSWD
# 收件人
receiver = sender


def ssl_mail():
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    server = smtplib.SMTP(smtp_server, smtp_port)
    # 只需要在创建SMTP对象后，立刻调用starttls()方法，就创建了安全连接。
    server.starttls()
    # 剩下的代码和前面的一模一样:
    server.set_debuglevel(1)


def test_mime_text():
    """
    测试MIMEText对象。
    :return:
    """
    from_addr = "me"
    to_addr = "you"

    msg = MIMEText('hello, send by Python...', 'plain', 'utf-8')

    # 邮件主题、如何显示发件人、收件人等信息并不是通过SMTP协议发给MTA，而是包含在发给MTA的文本中的，
    # 所以，必须把From、To和Subject添加到MIMEText中，才是一封完整的邮件。
    msg['From'] = _format_addr('Python爱好者 <{0}>'.format(from_addr))
    msg['To'] = _format_addr('管理员 <{0}>'.format(to_addr))
    msg['Subject'] = Header('来自SMTP的问候', 'utf-8').encode()

    send_email_message(msg.as_bytes())


def _format_addr(addr_str):
    """
    格式化地址字符串。
    :param addr_str:
    :return:
    """
    name, addr = parseaddr(addr_str)
    return formataddr((Header(name, 'utf-8').encode(), addr))


def test_multipart_mail():
    """
    测试MIMEMultipart对象。
    :return:
    """
    from_addr = "me"
    to_addr = "you"

    # 邮件对象
    msg = MIMEMultipart('alternative')
    msg['From'] = _format_addr('Python爱好者 <{0}>'.format(from_addr))
    msg['To'] = _format_addr('管理员 <{0}>'.format(to_addr))
    msg['Subject'] = Header('来自SMTP的问候', 'utf-8').encode()

    # 邮件正文是MIMEText,一次只能attach一个html页面
    # mime_text = MIMEText("""<html><body><h1>Hello</h1>
    #     <p>send by <a href="http://www.python.org">Python</a>...</p>
    #     </body></html>""", 'html', 'utf-8')
    # msg.attach(mime_text)
    # msg.attach(MIMEText('send with file...', 'plain', 'utf-8'))
    # msg.attach(MIMEText('<html><body><h1>Hello</h1></body></html>', 'html', 'utf-8'))

    # 正文-图片 只能通过html格式来放图片
    mail_msg = """<html><body>
        <p>html image</p>
        <p>截图如下：</p>
        <p><img src="cid:image1"></p>
        </body></html>
        """
    msg.attach(MIMEText(mail_msg, 'html', 'utf-8'))

    # 指定图片为当前目录
    img_path = os.path.join(image_dir, "demo1.png")
    fp = open(img_path, 'rb')
    msg_image = MIMEImage(fp.read())
    fp.close()
    # 定义图片ID，在HTML文本中引用
    msg_image.add_header('Content-ID', '<image1>')
    msg.attach(msg_image)

    # 添加附件就是加上一个MIMEBase，从本地读取一个图片。
    img_name = "lena.jpg"
    img_path = os.path.join(image_dir, img_name)
    with open(img_path, 'rb') as file:
        # 设置附件的MIME和文件名，这里是png类型:
        mime = MIMEBase('image', 'jpg', filename=img_name)
        # 加上必要的头信息:
        mime.add_header('Content-Disposition', 'attachment', filename=img_name)
        mime.add_header('Content-ID', '<0>')
        mime.add_header('X-Attachment-Id', '0')
        # 把附件的内容读进来:
        mime.set_payload(file.read())
        # 用Base64编码:
        encoders.encode_base64(mime)
        # 添加到MIMEMultipart:
        msg.attach(mime)

    c_type = 'application/octet-stream'
    maintype, subtype = c_type.split('/', 1)

    # 附件-文件
    txt_name = "logistic_test_set.txt"
    txt_path = os.path.join(os.path.join(resource_dir, "ml_data"), txt_name)
    file = MIMEBase(maintype, subtype)
    file.set_payload(open(txt_path, 'rb').read())
    file.add_header('Content-Disposition', 'attachment', filename='test.txt')
    encoders.encode_base64(file)
    msg.attach(file)

    # 发送邮件
    send_email_message(msg.as_bytes())


def test_simple_text_message():
    """
    测试EmailMessage的使用。发送simple text message。
    :return:
    """
    msg = EmailMessage()
    msg.set_content("Email Message Content")

    msg['Subject'] = "Email Message Subject"
    msg['From'] = "me"
    msg['To'] = "you"

    # 发送email信息
    send_email_message(msg.as_string())


def test_email_attachment():
    """
    测试发送附件。
    :return:
    """
    msg = EmailMessage()
    msg.set_content("Email With Image")

    msg['Subject'] = "Email Image Subject"
    msg['From'] = "me"
    msg['To'] = "you, she"

    # 增加图片附件
    img_name = "demo1.png"
    img_path = os.path.join(image_dir, img_name)
    with open(img_path, 'rb') as fp:
        img_data = fp.read()
        img_type = imghdr.what(None, img_data)
        msg.add_attachment(img_data, maintype='image',
                           subtype=img_type, filename=img_name)

    # 发送email信息
    send_email_message(msg.as_bytes())


def test_html_email():
    """
    测试html格式的email信息。
    :return:
    """
    msg = EmailMessage()
    msg['Subject'] = "Email Html Subject"
    msg['From'] = "me"
    msg['To'] = "you"

    # 添加文本内容
    msg.set_content("""\
        Hello Html Text Message Demo

        你好，这是基本的Email文本信息。
        """)

    # 添加html文本
    img_cid = make_msgid()
    msg.add_alternative("""\
    <html>
      <head></head>
      <body>
        <h1>html文本</h1>
        <p>Hello Html Text Message</p>
        <p>你好，这是基本的Email文本信息。</p>
        <img src="cid:{img_cid}" />
      </body>
    </html>
    """.format(img_cid=img_cid[1:-1]), subtype='html')

    # 增加图片
    img_name = "demo1.png"
    img_path = os.path.join(image_dir, img_name)
    with open(img_path, 'rb') as img:
        msg.get_payload()[1].add_related(img.read(), 'image', 'png',
                                         cid=img_cid)

    # 发送email信息
    send_email_message(msg.as_bytes())


def send_email_message(msg):
    """
    使用SMTP来发送信息。
    :param msg:
    :return:
    """
    smtp = smtplib.SMTP()
    smtp.connect(email_host, 25)
    smtp.set_debuglevel(2)
    smtp.login(sender, password)
    smtp.sendmail(sender, receiver, msg)
    smtp.quit()


if __name__ == "__main__":
    pass
    # test_simple_text_message()
    # test_email_attachment()
    # test_html_email()
    # test_mime_text()
    test_multipart_mail()


