#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 17:48:45 2018

@author: drtian
"""
import cv2
from hungary import center

import configparser
import time
from alarm_post import Alarm

config=configparser.ConfigParser()  
config.read(u'config.ini') 



def bbOverLap (box1,box2):
    if box1[1] > box2[3]  :
        return 0.0
    if box1[0] > box2[2]  :
        return 0.0
    if box1[3] < box2[1] :
        return 0.0
    if box1[2]  < box2[0] :
        return 0.0
    colInt = min(box1[3],box2[3]) - max(box1[1],box2[1])
    rowInt = min(box1[2] , box2[2]) - max( box1[0], box2[0])
    intersection = colInt * rowInt 
    return intersection/((box2[3]-box2[1])*(box2[2]-box2[0]))


class Pedestrian():

    def __init__(self,id,frame,track_window,CAMERA_ID):
        self.id = int(id)
#        ymin,xmin,ymax,xmax = track_window
        self.center = center(track_window)
        self.track_window = track_window
        self.age = 0
        self.totalVisionbleCount = 0 
        self.consecutiveInvisibleCount = 0
        self.helmetInvisibleCount = 0
        self.tracker = cv2.TrackerBoosting_create()
        ymin,xmin,ymax,xmax =track_window
        bbox1 = (xmin,ymin,xmax-xmin,ymax-ymin)
        ok = self.tracker.init(frame, bbox1)
        self.ALARM_LEVEL = int(config.get('ALARM','ALARM_LEVEL')) 
        self.ALARM_PERIOD = int(config.get('ALARM','ALARM_PERIOD'))
        self.pre_alarm_age = 0
        self.camera_id = CAMERA_ID
    def __del__(self):
        print ("Pedestrian %d destroyed" % self.id)
        return(self.id)

    def update(self, frame):
        print( "updating %d " % self.id)
#        print("self.pre_alarm_age , self.age :" + str(self.pre_alarm_age) +str(self.age))
        self.age +=1
        
        ok,bbox = self.tracker.update(frame)
        xmin,ymin,xmax,ymax =  int(bbox[0]), int(bbox[1]),int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
        if ok :
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        
        self.track_window = [ymin,xmin,ymax,xmax]
        self.center = center(self.track_window )
       
        size = frame.shape
        if int(self.center[0]) < size[1]*0.1 or int(self.center[0]) > size[1]*0.9 or int(self.center[1]) > size[0]*0.8 or int(self.center[1]) < size[0]*0.2:
            self.__del__()
            return(True,self.id,self.track_window)
            
        if int(self.consecutiveInvisibleCount) > 5:
            self.__del__()
            return(True,self.id,self.track_window)
        cv2.circle(frame, (int(self.center[0]), int(self.center[1])), 4, (255, 0, 0), -1)
        cv2.putText(frame, "ID: %d| age: %d| center : %s Count 1: %d | Count 2: -> %d" % (self.id, self.age,self.center,self.consecutiveInvisibleCount,self.helmetInvisibleCount), (10, (self.id + 1) * 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (0, 255, 0),
        1,
        cv2.LINE_AA)
        
        if self.helmetInvisibleCount  > self.ALARM_LEVEL :
#            and (self.helmetInvisibleCount - self.pre_alarm_age) >= self.ALARM_PERIOD:
            self.ALARM_LEVEL += self.ALARM_PERIOD
            cv2.putText(frame, "wairning : helmet ! ID: %d -> %s   level: -> %d pre_alarm_age: %d " % (self.id, self.center,self.helmetInvisibleCount,self.pre_alarm_age), (640, (self.id + 1) * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA)

            start_time = time.localtime(time.time())
            end_time = time.localtime(time.time())
            alarmImg = cv2.resize(frame,(960, 540),interpolation=cv2.INTER_CUBIC)
            _, imagefile = cv2.imencode('.jpg', alarmImg) 
            resp = Alarm().addAlarm(str(self.camera_id),  'helmet_onhead', time.strftime('%Y-%m-%d %H:%M:%S',start_time), time.strftime('%Y-%m-%d %H:%M:%S',end_time), imagefile)
#            imagefile = open(str(time.strftime('%Y-%m-%d-%H-%M-%S ',start_time))+ 'warning.jpg', 'rb')
#            resp = Alarm().addAlarm(str(self.camera_id),  'helmet_onhead', time.strftime('%Y-%m-%d %H:%M:%S',start_time), time.strftime('%Y-%m-%d %H:%M:%S',end_time), imagefile.read())
            print(resp)
#            imagefile.close()
#            shutil.move(str(time.strftime('%Y-%m-%d-%H-%M-%S ',start_time))+ 'warning.jpg','warning')
        return(False,self.id,self.track_window)
    
    def correct(self,frame,track_window):
        # TODO : tracker release
        self.tracker = cv2.TrackerBoosting_create()
        ymin,xmin,ymax,xmax =track_window
        bbox1 = (xmin,ymin,xmax-xmin,ymax-ymin)
        ok = self.tracker.init(frame, bbox1)
        self.center = center(track_window)
        self.track_window = track_window
    def invisible(self):
        self.consecutiveInvisibleCount += 1
         
    def helmet_detection(self,helmet_boxes):
        helmet_f = True
        if len(helmet_boxes)>0:
            box1 = self.track_window
            
            for box2 in helmet_boxes:
                overlap = bbOverLap(box1,box2)
                if overlap >=0.5:
                    print('overlap :' + str(overlap))
                    helmet_f = False
        if helmet_f :
            self.helmetInvisibleCount += 1
        else:
            self.helmetInvisibleCount = 0
            self.ALARM_LEVEL = int(config.get('ALARM','ALARM_LEVEL'))

        
