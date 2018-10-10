#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 17:48:45 2018

@author: drtian
"""
import cv2
from hungary import center

font = cv2.FONT_HERSHEY_SIMPLEX

class Pedestrian():
    
    def _init_(self,id,frame,track_window):
        self.id = int(id)
#        ymin,xmin,ymax,xmax = track_window
        self.center = center(track_window)
        self.track_window = track_window
        self.age = 0
        self.totalVisionbleCount = 0 
        self.consecutiveInvisibleCount = 0
        self.tracker = cv2.TrackerBoosting_create()
    
    def __del__(self):
        print ("Pedestrian %d destroyed" % self.id)

    def update(self, frame):
        print( "updating %d " % self.id)
        ok,bbox = self.tracker.update(frame)
        xmin,ymin,xmax,ymax =  int(bbox[0]), int(bbox[1]),int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
        if ok :
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        
        self.track_window = [ymin,xmin,ymax,xmax]
        self.center = center(self.track_window )
        cv2.putText(frame, "ID: %d -> %s" % (self.id, self.center), (10, (self.id + 1) * 25),
        font, 0.6,
        (0, 255, 0),
        1,
        cv2.LINE_AA)
#  def update(self, frame):
#    # print "updating %d " % self.id
#    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#    back_project = cv2.calcBackProject([hsv],[0], self.roi_hist,[0,180],1)
#    
#    if args.get("algorithm") == "c":
#      ret, self.track_window = cv2.CamShift(back_project, self.track_window, self.term_crit)
#      pts = cv2.boxPoints(ret)
#      pts = np.int0(pts)
#      self.center = center(pts)
#      cv2.polylines(frame,[pts],True, 255,1)
#      
#    if not args.get("algorithm") or args.get("algorithm") == "m":
#      ret, self.track_window = cv2.meanShift(back_project, self.track_window, self.term_crit)
#      x,y,w,h = self.track_window
#      self.center = center([[x,y],[x+w, y],[x,y+h],[x+w, y+h]])  
#      cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 0), 2)
#
#    self.kalman.correct(self.center)
#    prediction = self.kalman.predict()
#    cv2.circle(frame, (int(prediction[0]), int(prediction[1])), 4, (255, 0, 0), -1)
#    # fake shadow
#    cv2.putText(frame, "ID: %d -> %s" % (self.id, self.center), (11, (self.id + 1) * 25 + 1),
#        font, 0.6,
#        (0, 0, 0),
#        1,
#        cv2.LINE_AA)
#    # actual info
#    cv2.putText(frame, "ID: %d -> %s" % (self.id, self.center), (10, (self.id + 1) * 25),
#        font, 0.6,
#        (0, 255, 0),
#        1,
#        cv2.LINE_AA)