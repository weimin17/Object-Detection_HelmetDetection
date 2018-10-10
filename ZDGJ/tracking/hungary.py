#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 17:03:18 2018

@author: drtian
"""

from  scipy.optimize import linear_sum_assignment

import numpy as np

from scipy.spatial import distance

import pandas as pd



def center(points):
    """calculates centroid of a given matrix"""
    x = (points[1] + points[3] ) / 2
    y = (points[0] + points[2]) / 2
    return np.array([np.float32(x), np.float32(y)], np.float32)



#cost = np.array([[4,1,3],[2,0,5],[3,2,2]])

#boxes = {0: ([390.,  29., 838., 234.]),
#         1:([390.,  29., 838., 234.]),
#         3:([390.,  29., 838., 234.])}
#
#boxes = {0: ([390.,  29., 838., 234.]),
#         1:([390.,  29., 838., 234.]),
#         3:([390.,  29., 838., 234.])}


#pedestrian_boxes = np.array([[220., 680., 327., 747.],
#       [197., 805., 292., 857.],
#       [216., 736., 319., 807.]], dtype=np.float32)



#{0: array([208., 745., 309., 805.], dtype=float32),
# 1: array([192., 807., 292., 857.], dtype=float32)}
#array([[197., 808., 291., 858.],
#       [214., 742., 317., 815.],
#       [216., 695., 325., 760.]], dtype=float32)


def detection_to_track_assignment(boxes,pedestrian_boxes,size = (1280, 720) ,cost_of_non_assignment = 0.3):
    
    nTracks = len(boxes)
    nDtections = len(pedestrian_boxes)
    boxes_key = []
    boxes_box = []
    for k,v in boxes.items():
        ymin,xmin,ymax,xmax = v
        ymin,xmin,ymax,xmax = ymin/size[1],xmin/size[0],ymax/size[1],xmax/size[0]
        bbox1 = (ymin,xmin,ymax,xmax)
        boxes_key.append(k) 
        boxes_box.append(bbox1)
    pedestrian_boxes_norm = []
    for i,box in enumerate(pedestrian_boxes):
#        print(box)
        bbox1 = (box[0]/size[1],box[1]/size[0],box[2]/size[1],box[3]/size[0])
        pedestrian_boxes_norm.append(bbox1) 
         
    boxes_box = np.array(list(map(center,boxes_box)))
    pedestrian_boxes_norm = np.array(list(map(center,pedestrian_boxes_norm)))
    cost = distance.cdist(boxes_box,pedestrian_boxes_norm,'euclidean')
    row_ind,col_ind = linear_sum_assignment(cost)

    indx = []
    for i,x in enumerate(row_ind):
        if cost[x][col_ind[i]] <=  cost_of_non_assignment:
            indx.append(i)
    row_ind,col_ind = row_ind[indx],col_ind[indx]        
    assignments = dict()            
    for i,x in enumerate(row_ind):
        assignments[boxes_key[x]] = col_ind[i]
        
    tracks = pd.Series(np.arange(nTracks),dtype=np.int)
    boxes_key = np.array(boxes_key)
    unassignedTracks = boxes_key[~tracks.isin(row_ind)]
    
    Dtections = pd.Series(np.arange(nDtections),dtype=np.int)
    unassignedDetections = Dtections[~Dtections.isin(col_ind)]
    
    return assignments, unassignedTracks, unassignedDetections
 