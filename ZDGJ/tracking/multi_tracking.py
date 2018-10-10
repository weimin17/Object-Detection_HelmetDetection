#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 11:27:27 2018

@author: drtian
"""

import cv2
import sys
 
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.') 
if __name__ == '__main__' :
 
    # Set up tracker.
    # Instead of MIL, you can also use
 
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[0]
    
    
    tracker = cv2.MultiTracker_create()
    
#    if int(minor_ver) < 3:
#        tracker = cv2.Tracker_create(tracker_type)
#    else:
#        if tracker_type == 'BOOSTING':
#            tracker = cv2.TrackerBoosting_create()
#        if tracker_type == 'MIL':
#            tracker = cv2.TrackerMIL_create()
#        if tracker_type == 'KCF':
#            tracker = cv2.TrackerKCF_create()
#        if tracker_type == 'TLD':
#            tracker = cv2.TrackerTLD_create()
#        if tracker_type == 'MEDIANFLOW':
#            tracker = cv2.TrackerMedianFlow_create()
#        if tracker_type == 'GOTURN':
#            tracker = cv2.TrackerGOTURN_create()

    # Read video
    video = cv2.VideoCapture("/home/drtian/ZDGJ/监控录像/1#机组脱硫CESM小室1529636867_2CE8C22B/未命名项目.mp4")
 
    # Exit if video not opened.
    if not video.isOpened():
        print ("Could not open video")
        sys.exit()
 
    # Read first frame.
    ok, frame = video.read()
    for i in range(150):
        ok, frame = video.read()
    if not ok:
        print ('Cannot read video file')
        sys.exit()
     
    # Define an initial bounding box
    bbox = (287, 23, 86, 320)
 
    # Uncomment the line below to select a different bounding box
#    bbox = cv2.selectROI(frame, False)
 
    bbox1 = cv2.selectROI('tracking1', frame)
    bbox2 = cv2.selectROI('tracking2', frame)
    bbox3 = cv2.selectROI('tracking3', frame)
    ok = tracker.add(cv2.TrackerKCF_create(), frame, bbox1)
    ok = tracker.add(cv2.TrackerKCF_create(), frame, bbox2)
    ok = tracker.add(cv2.TrackerKCF_create(), frame, bbox3)
    # Initialize tracker with first frame and bounding box
 
    while True:
        # Read a new frame
        ok, frame = video.read()
        frame = cv2.UMat(frame)
        if not ok:
            break
         
        # Start timer
        timer = cv2.getTickCount()
 
        # Update tracker
        ok, boxes = tracker.update(frame)
        print (ok, boxes)

        for newbox in boxes:
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(frame, p1, p2, (200,0,0))
 
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
 
        # Draw bounding box
#        if ok:
#            # Tracking success
#            p1 = (int(bbox[0]), int(bbox[1]))
#            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
#            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
#        else :
#            # Tracking failure
#            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
# 
#        # Display tracker type on frame
#        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
     
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
 
        # Display result
        cv2.imshow("Tracking", frame)
 
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break
    
cv2.destroyAllWindows()
    
    
    
