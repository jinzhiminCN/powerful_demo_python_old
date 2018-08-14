# -*- coding:utf-8 -*-

# ==============================================================================
# 测试文件使用的相关方法。
# ==============================================================================
import os
import zipfile
import tarfile
import shutil
import config.common_config as com_config

# =========================== function ===========================


def make_zip(src_dir, zip_filename):
    """
    打包目录为zip文件
    :param src_dir:
    :param zip_filename:
    :return:
    """
    zipf = zipfile.ZipFile(zip_filename, 'w', compression=zipfile.zlib.DEFLATED)
    pre_len = len(os.path.dirname(src_dir))
    for parent, dir_names, file_names in os.walk(src_dir):
        for filename in file_names:
            file_path = os.path.join(parent, filename)
            # 相对路径
            arc_name = file_path[pre_len:].strip(os.path.sep)
            zipf.write(file_path, arc_name)
    zipf.close()


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
    print("Start to unzip file %s to folder %s ..." % (zip_filename, unzip_dirname))

    # Check input ...
    if not os.path.exists(full_zip_filename):
        print("Dir/File %s is not exist, Press any key to quit..." % full_zip_filename)
        input_str = input()
        return

    if not os.path.exists(full_unzip_dirname):
        os.mkdir(full_unzip_dirname)
    else:
        if os.path.isfile(full_unzip_dirname):
            print("File %s is exist, are you sure to delet it first ? [Y/N]" % full_unzip_dirname)
            while 1:
                input_str = input()
                if input_str == "N" or input_str == "n":
                    return
                else:
                    if input_str == "Y" or input_str == "y":
                        os.remove(full_unzip_dirname)
                        print("Continue to unzip files ...")
                        break

    # Start extract files ...
    srcZip = zipfile.ZipFile(full_unzip_dirname, "r")
    for eachfile in srcZip.namelist():
        if eachfile.endswith('/'):
            # is a directory
            print('Unzip directory %s ...' % each_filename)

            os.makedirs(os.path.normpath(os.path.join(full_unzip_dirname, eachfile)))
            continue
        print("Unzip file %s ..." % eachfile)

        each_filename = os.path.normpath(os.path.join(full_unzip_dirname, eachfile))
        each_dirname = os.path.dirname(each_filename)
        if not os.path.exists(each_dirname):
            os.makedirs(each_dirname)
        fd = open(each_filename, "wb")
        fd.write(srcZip.read(eachfile))
        fd.close()
    srcZip.close()
    print("Unzip file succeed!")


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

