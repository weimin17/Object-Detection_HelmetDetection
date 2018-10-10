#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
对比两个文件夹中文件个数（后缀不同），并删除多余的文件。
仅限于文件夹下没有子文件夹，只有文件
'''

import os
import re
import sys


def get_CurName(path):
    cur_Files = os.listdir(path)
    files_without_suffix = []
    for file_Name in cur_Files:
        files_without_suffix.append(re.split(r'[.]', file_Name)[0])
    print('Files number in ' + path + 'is : ' + str(len(files_without_suffix)))
    return files_without_suffix


def del_MisFile(path1, path2):
    FileList1 = get_CurName(path1)
    FileList2 = get_CurName(path2)
    del_files = []
    for file_Name in FileList1:
        if file_Name in FileList2:
            pass
            # print(file_Name)
        else:
            del_path = os.path.join(path1, (file_Name + '.xml'))
            del_files.append(del_path)
            print(del_path)
            os.remove(del_path)
    print('del number:' + str(len(del_files)))
    return del_files


if __name__ == '__main__':
    path1 = os.path.join(os.getcwd(), 'JPEGImages')
    path2 = os.path.join(os.getcwd(), 'Annotations')
    del_MisFile(path2, path1)
