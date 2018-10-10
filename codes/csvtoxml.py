#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
csv to xml
'''

import os,cv2
import re
import pandas as pd
import shutil
from xml import etree
from xml.etree import ElementTree
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, Comment, ElementTree

def loadcsv(file_Name):
    df = pd.read_csv(file_Name,header = None)
    # convert pandas frame to narry
    pd2narry = df.ix[:,:].values
    return pd2narry


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = etree.ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


# from ElementTree_pretty import prettify
def buildxml(Pic_name,xmin,ymin,xmax,ymax,weight_num,height_num):
    top = Element('annotation')
    
    comment = Comment('Generated for BataFoundation - louis0815')
    top.append(comment)
    
    filename = SubElement(top, 'filename')
    filename.text = str(Pic_name)
    sizePic = SubElement(top,'size')
    weight = SubElement(sizePic,'width')
    height = SubElement(sizePic,'height')
    depth = SubElement(sizePic,'depth')
    weight.text = '%d'%(weight_num)
    height.text = '%d'%(height_num)
    depth.text = '3'
    objectname = SubElement(top, 'object')
    name = SubElement(objectname,'name')
    name.text = 'helmet'
    bndbox = SubElement(objectname,'bndbox')
    elexmin = SubElement(bndbox,'xmin')
    elexmin.text = xmin
    eleymin = SubElement(bndbox,'ymin')
    eleymin.text = ymin
    elexmax = SubElement(bndbox,'xmax')
    elexmax.text = xmax
    eleymax = SubElement(bndbox,'ymax')
    eleymax.text = ymax

    xmlfilepath = str('/Users/insisterlouis/Workspaces/HardhatDetector/data/test/Annotations/'+(re.split(r'[.]', Pic_name)[0])+'.xml')

    xml_object = minidom.parseString(etree.ElementTree.tostring(top,encoding="utf-8")).toprettyxml(indent=" ")
    with open(xmlfilepath, "wb") as writter:
        writter.write(xml_object)
    return top


def buildxmlwithoutelement(pic_name,weight_num,height_num):
    top = Element('annotation')
    
    comment = Comment('Generated for BataFoundation - louis0815')
    top.append(comment)
    
    filename = SubElement(top, 'filename')
    filename.text = str(pic_name)
    sizePic = SubElement(top,'size')
    weight = SubElement(sizePic,'width')
    height = SubElement(sizePic,'height')
    depth = SubElement(sizePic,'depth')
    weight.text = '%d'%(weight_num)
    height.text = '%d'%(height_num)
    depth.text = '3'
    
    xmlfilepath = str('/Users/insisterlouis/Workspaces/HardhatDetector/data/test/Annotations/'+(re.split(r'[.]', pic_name)[0])+'.xml')

    xml_object = minidom.parseString(etree.ElementTree.tostring(top,encoding="utf-8")).toprettyxml(indent="   ")
    with open(xmlfilepath, "wb") as writter:
        writter.write(xml_object)
        
    return top


def addxmlelement(pic_name,xmin,ymin,xmax,ymax):

    # parse an xml file by name
    xmlfilepath = str('/Users/insisterlouis/Workspaces/HardhatDetector/data/test/Annotations/'+(re.split(r'[.]', pic_name)[0])+'.xml')
    prexml = minidom.parse(xmlfilepath)
    
    prexml = etree.ElementTree.parse('/Users/insisterlouis/Workspaces/HardhatDetector/data/test/Annotations/'+(re.split(r'[.]', pic_name)[0])+'.xml')  
    root = prexml.getroot()
    
    # # adding an element to the root node
    # element = root.makeelement('objectaaaa')  
    # root.extend(element)
    
    # adding an element to the seconditem node
    attrib = {}  
    # subelement = root[0][1].makeelement('seconditem', attrib)  
    objectname = etree.ElementTree.SubElement(root, 'object', attrib)

    # objectname = SubElement(items, 'object')
    name = SubElement(objectname,'name')
    name.text = 'helmet'
    bndbox = SubElement(objectname,'bndbox')
    elexmin = SubElement(bndbox,'xmin')
    elexmin.text = xmin
    eleymin = SubElement(bndbox,'ymin')
    eleymin.text = ymin
    elexmax = SubElement(bndbox,'xmax')
    elexmax.text = xmax
    eleymax = SubElement(bndbox,'ymax')
    eleymax.text = ymax

    # prexml.write(xmlfilepath)
    xml_object = minidom.parseString(etree.ElementTree.tostring(root,encoding="utf-8")).toprettyxml(indent="")
    with open(xmlfilepath, "wb") as writter:
        writter.write(xml_object)
    return prexml
    

def creatxml(file_Name,PATH_TO_ORI_PIC):
    if not os.path.exists("/Users/insisterlouis/Workspaces/HardhatDetector/data/test/Annotations/"):  
        os.makedirs("/Users/insisterlouis/Workspaces/HardhatDetector/data/test/Annotations/")
    else:
        # 递归删除非空文件夹
        shutil.rmtree("/Users/insisterlouis/Workspaces/HardhatDetector/data/test/Annotations/")
        os.makedirs("/Users/insisterlouis/Workspaces/HardhatDetector/data/test/Annotations/")
    pd2narry = loadcsv(file_Name)
    for index, record in enumerate(pd2narry):
        img_path = os.path.join(PATH_TO_ORI_PIC,record[0])
        img = cv2.imread(img_path)
        size = img.shape
        weight = int(size[1])
        height = int(size[0])
        if pd2narry[index][0]==pd2narry[index-1][0]:
            coordinate = re.split(r' ',record[1])
            xmlobject = addxmlelement(record[0],coordinate[0],coordinate[1],coordinate[2],coordinate[3])

        else:    
            if pd.isnull(record[1]):
                xmlobject = buildxmlwithoutelement(record[0],weight,height)
            else:
                coordinate = re.split(r' ',record[1])
                xmlobject = buildxml(record[0],coordinate[0],coordinate[1],coordinate[2],coordinate[3],weight,height)

if __name__ == '__main__':
    file_Name = '/Users/insisterlouis/Workspaces/HardhatDetector/data/bndbox_0.7_train-test_faster_RCNN_Inception_ResNet_v2_0814.csv'
    PATH_TO_ORI_PIC = '/Users/insisterlouis/Workspaces/HardhatDetector/data/test/JPEGImages/'
    creatxml(file_Name,PATH_TO_ORI_PIC)
