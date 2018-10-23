# -*- coding:utf-8 -*-

# ==============================================================================
# 测试文件使用的相关方法。
# ==============================================================================
import os
import zipfile
import tarfile
import shutil
import config.common_config as com_config
from util.log_util import LoggerUtil

# 日志器
common_logger = LoggerUtil.get_common_logger()


def make_zip(src_dir, zip_filename):
    """
    打包目录为zip文件
    :param src_dir:
    :param zip_filename:
    :return:
    """
    zip_f = zipfile.ZipFile(zip_filename, 'w', compression=zipfile.zlib.DEFLATED)
    pre_len = len(os.path.dirname(src_dir))
    for parent, dir_names, file_names in os.walk(src_dir):
        for filename in file_names:
            file_path = os.path.join(parent, filename)
            # 相对路径
            arc_name = file_path[pre_len:].strip(os.path.sep)
            zip_f.write(file_path, arc_name)
    zip_f.close()


def make_targz(output_filename, src_dir):
    """
    # 一次性打包整个根目录。空子目录会被打包。
    # 如果只打包不压缩，将"w:gz"参数改为"w:"或"w"即可。
    :param output_filename:
    :param src_dir:
    :return:
    """
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(src_dir, arcname=os.path.basename(src_dir))


def make_targz_one_by_one(output_filename, src_dir):
    """
    # 逐个添加文件打包，未打包空子目录。可过滤文件。
    # 如果只打包不压缩，将"w:gz"参数改为"w:"或"w"即可。
    :param output_filename:
    :param src_dir:
    :return:
    """
    tar = tarfile.open(output_filename, "w:gz")
    for root, sub_dirs, files in os.walk(src_dir):
        for file in files:
            file_path = os.path.join(root, file)
            tar.add(file_path)
    tar.close()


def unzip_dir(zip_filename, unzip_dirname):
    """
    解压缩一个文件。
    :param zip_filename:
    :param unzip_dirname:
    :return:
    """
    full_zip_filename = os.path.abspath(zip_filename)
    full_unzip_dirname = os.path.abspath(unzip_dirname)
    common_logger.info("Start to unzip file {0} to folder {1} ...".format(zip_filename, unzip_dirname))

    # Check input ...
    if not os.path.exists(full_zip_filename):
        common_logger.info("Dir/File {0} is not exist, Press any key to quit...".format(full_zip_filename))
        input_str = input()
        return

    if not os.path.exists(full_unzip_dirname):
        os.mkdir(full_unzip_dirname)
    else:
        if os.path.isfile(full_unzip_dirname):
            common_logger.info("File {0} is exist, are you sure to delet it first ? [Y/N]".format(full_unzip_dirname))
            while 1:
                input_str = input()
                if input_str == "N" or input_str == "n":
                    return
                else:
                    if input_str == "Y" or input_str == "y":
                        os.remove(full_unzip_dirname)
                        common_logger.info("Continue to unzip files ...")
                        break

    # Start extract files ...
    src_zip = zipfile.ZipFile(full_unzip_dirname, "r")
    for each_file in src_zip.namelist():
        if each_file.endswith('/'):
            # is a directory
            common_logger.info('Unzip directory {0} ...'.format(each_filename))

            os.makedirs(os.path.normpath(os.path.join(full_unzip_dirname, each_file)))
            continue
        common_logger.info("Unzip file {0} ...".format(each_file))

        each_filename = os.path.normpath(os.path.join(full_unzip_dirname, each_file))
        each_dirname = os.path.dirname(each_filename)
        if not os.path.exists(each_dirname):
            os.makedirs(each_dirname)
        fd = open(each_filename, "wb")
        fd.write(src_zip.read(each_file))
        fd.close()
    src_zip.close()
    common_logger.info("Unzip file succeed!")


def test_file_archive():
    """
    测试文件操作。
    :return:
    """
    src_dir = os.path.join(com_config.RESOURCE_DIR, "image_data")
    zip_filename = os.path.join(com_config.RESOURCE_DIR, "image_data.zip")
    dst_dir = os.path.join(com_config.RESOURCE_DIR, "image_data")
    # make_zip(src_dir, zip_filename)
    ret = shutil.make_archive(dst_dir, "gztar", root_dir=src_dir)


if __name__ == "__main__":
    test_file_archive()
    pass

