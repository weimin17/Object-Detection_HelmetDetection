#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
pick up pictures without helmet, for ZDGJ project
'''

import os
import shutil
import re
import pandas as pd


def loadcsv(file_Name):
    df = pd.read_csv(file_Name)
    # convert pandas frame to narry
    pd2narry = df.ix[:,:].values
    return pd2narry


def copyfile(file_Name):
    if not os.path.exists("JPEGImages_picked"):  
        os.makedirs("JPEGImages_picked")
    else:
        # 递归删除非空文件夹
        shutil.rmtree("JPEGImages_picked")
        os.makedirs("JPEGImages_picked")
    pd2narry = loadcsv(file_Name)
    for index, record in enumerate(pd2narry):
    
        if pd.isnull(record[1]):
            Picfilepath = str('JPEGImages/'+(re.split(r'[.]', str(record[0]))[0])+'.jpg')
            print(Picfilepath)

            targetFile = str('JPEGImages_picked/'+(re.split(r'[.]', record[0])[0])+'.jpg')
            shutil.copyfile(Picfilepath,targetFile)


if __name__ == '__main__':
    file_Name = 'train_labels.csv'
    copyfile(file_Name)
