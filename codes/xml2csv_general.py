#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
xml to csv
'''


import xml.etree.ElementTree as ET
import csv,os,re


def line2csv(path):
    cur_Files = os.listdir(path)
    # open a file for writing
    helmet_data = open('/Users/insisterlouis/Workspaces/HardhatDetector/data/0816_1.csv', 'w')

    # create the csv writer object
    csvwriter = csv.writer(helmet_data)
    for file_Name in cur_Files:
        file_Name = re.split(r'[.]', file_Name)[0]
        tree = ET.parse(os.path.join("/Users/insisterlouis/Workspaces/HardhatDetector/data/test/Annotations/"+file_Name+".xml"))
        # print(os.path.join("Annotations/"+file_Name+".xml"))
        root = tree.getroot()
        if not root.findall('object'):
            resident = [file_Name+".jpg",""]
            print(resident)
            csvwriter.writerow(resident)
        else:
            for member in root.findall('object'):
                if len(member)==0:
                    print('nonononononon')
                    resident = [file_Name+".jpg",""]
                    print(resident)
                    csvwriter.writerow(resident)
                else:
                    resident = []
                    bndbox = []
                    xmin = int(member.find('bndbox').find('xmin').text)
                    ymin = int(member.find('bndbox').find('ymin').text)
                    xmax = int(member.find('bndbox').find('xmax').text)
                    ymax = int(member.find('bndbox').find('ymax').text)
                    csvwriter.writerow([file_Name+".jpg","%s %s %s %s"%(xmin,ymin,xmax,ymax)])

    helmet_data.close()

if __name__ == '__main__':
    path = "/Users/insisterlouis/Workspaces/HardhatDetector/data/test/Annotations"

    line2csv(path)
