# -*- coding:utf-8 -*-

# ==============================================================================
# 测试protobuf的基本用法。
# ==============================================================================
from middleware_basic.protobuf_basic import person_pb2
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()


def test_protobuf():
    """
    测试protobuf。
    :return:
    """
    person = person_pb2.Person()

    person.id = 1
    person.name = "mike"
    person.email = "mike@qq.com"

    phone_number = person.phone.add()
    phone_number.number = "123456"
    phone_number.type = person_pb2.Person.MOBILE

    # 序列化
    serialize_person = person.SerializeToString()
    common_logger.info("serialize: {0}, type: {1}".format(serialize_person, type(serialize_person)))

    # 反序列化
    person.ParseFromString(serialize_person)
    common_logger.info("p_id: {0}, p_name: {1}, p_email: {2}"
                       .format(person.id, person.name, person.email))

    for phone_number in person.phone:
        common_logger.info("phone_number: {0}, phone_type: {1}"
                           .format(phone_number.number, phone_number.type))


if __name__ == "__main__":
    test_protobuf()
    pass
