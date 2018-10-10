#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 10:21:18 2018

@author: drtian
"""

import configparser  
import sys
config=configparser.ConfigParser()  
config.read(u'/home/drtian/ZDGJ/tracking/config.ini')  

config.add_section("book")  
config.set("book","title","这是标题")  
config.set("book","author","大头爸爸")  
config.add_section("size")  
config.set("size","size",str(1024))  
config.write(open(u'/home/drtian/ZDGJ/tracking/config.ini','a'))  
