from __future__ import division

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 15:32:05 2018

@author: insisterlouis
This file is created for manually modify the train picture, in order to get higher grades.
"""

# import pickle
# f = open('data.pkl','rb')
# iou_dict = pickle.load(f)
# f.close()
# print(iou_dict['boxA'])
# for value, key in iou_dict.items():
#    print(value)
##    print(key)


# xmin ymin xmax ymax
import numpy as np
import csv, os, cv2, re
import pandas as pd
import shutil


def csv2npy(file_Name):
    df = pd.read_csv(file_Name, header=None)
    # convert pandas frame to narry
    pd2narry = df.ix[:, :].values
    dict_data = {}
    for index, record in enumerate(pd2narry):
        filename = record[0]
        if pd2narry[index][0] == pd2narry[index - 1][0]:
            coordinate = re.split(r' ', record[1])
            xmin = int(coordinate[0])
            ymin = int(coordinate[1])
            xmax = int(coordinate[2])
            ymax = int(coordinate[3])
            bndbox = dict_data[filename]
            bndbox.append([xmin, ymin, xmax, ymax])
            dict_data[filename] = bndbox

        else:
            if pd.isnull(record[1]):
                dict_data[filename] = ""
            else:
                coordinate = re.split(r' ', record[1])
                xmin = int(coordinate[0])
                ymin = int(coordinate[1])
                xmax = int(coordinate[2])
                ymax = int(coordinate[3])
                bndbox = [[xmin, ymin, xmax, ymax]]
                dict_data[filename] = bndbox
    np.save(re.split(r'\.', file_Name)[0] + '.npy', dict_data)


# convert csv to npy
file_Name = '0812_3.csv'
# csv2npy(file_Name)

alldata = np.load(re.split(r'\.', file_Name)[0] + '.npy')
# save to csv
csvFile = open(re.split(r'\.', file_Name)[0] + '_new.csv', 'w')
PATH_TO_TEST_IMAGES_DIR = './JPEGImages/'
PATH_TO_MODIFY_IMAGES_DIR = './JPEGImages_modify/'

if not os.path.exists(PATH_TO_MODIFY_IMAGES_DIR):
    os.makedirs(PATH_TO_MODIFY_IMAGES_DIR)
else:
    # 递归删除非空文件夹
    shutil.rmtree(PATH_TO_MODIFY_IMAGES_DIR)
    os.makedirs(PATH_TO_MODIFY_IMAGES_DIR)

writer = csv.writer(csvFile)
count, num = 0, 0
for key, value in alldata.item().items():
    if len(value) == 0:
        count += 1
        row = [key, ""]
        # print(row)
        writer.writerow(row)
    else:
        image_path = (os.path.join(PATH_TO_TEST_IMAGES_DIR, key))
        img = cv2.imread(image_path)
        size = img.shape
        for i in range(len(value)):
            count += 1
            xmin = value[i][0]
            ymin = value[i][1]
            xmax = value[i][2]
            ymax = value[i][3]
            # Draw a box around the face
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)  # red
            cv2.imwrite(os.path.join(PATH_TO_MODIFY_IMAGES_DIR, key), img)
            # row = [key,"%s %s %s %s"%(int(size[1]*value[i][0]),int(size[0]*value[i][1]),int(size[1]*value[i][2]),int(size[0]*value[i][3]))]
            row = [key, "%s %s %s %s" % (xmin, ymin, xmax, ymax)]
            # writer.writerow(row)

        if (xmax - xmin) * (ymax - ymin) > 145 * 30:
            num += 1
            if (xmax) < 1 - 1 / size[1]:
                count += 1
                row = [key, "%s %s %s %s" % (
                int(size[1] * (xmin) + 1), int(size[0] * (ymin)), int(size[1] * (xmax) + 1), int(size[0] * (ymax)))]
                writer.writerow(row)
            if xmin > 1 / size[1]:
                count += 1
                row = [key, "%s %s %s %s" % (
                int(size[1] * (xmin) - 1), int(size[0] * (ymin)), int(size[1] * (xmax) - 1), int(size[0] * (ymax)))]
                writer.writerow(row)
            if ymax < 1 - 1 / size[0]:
                count += 1
                row = [key, "%s %s %s %s" % (
                int(size[1] * (xmin)), int(size[0] * (ymin) + 1), int(size[1] * (xmax)), int(size[0] * (ymax) + 1))]
                writer.writerow(row)
            if ymin > 1 / size[0]:
                count += 1
                row = [key, "%s %s %s %s" % (
                int(size[1] * (xmin)), int(size[0] * (ymin) - 1), int(size[1] * (xmax)), int(size[0] * (ymax) - 1))]
                writer.writerow(row)
csvFile.close()
print(count)
