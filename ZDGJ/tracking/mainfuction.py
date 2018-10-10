#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 13:57:45 2018

@author: drtian
"""

import numpy as np
import tensorflow as tf
import cv2
from hungary import detection_to_track_assignment
from Pedestrian_without_alarm import Pedestrian
from detector import detect_for_single_img
import os
import time

def test_main(CAMERA_IP,
              DETECT_FREQUECY = 25,
              GPU_MEMORY = 0.7,
              PATH_TO_CKPT = '/home/drtian/ZDGJ/0802_data_fasterRCNN_inception_k2office_finetune/frozen_inference_graph.pb'):


    CAMERA_ID = 5

    videoCapture = cv2.VideoCapture(CAMERA_IP)

#    CAMERA_ID = FLAGS.CAMERA_ID
#    DETECT_FREQUECY = FLAGS.DETECT_FREQUECY
#    videoCapture = cv2.VideoCapture(FLAGS.CAMERA_IP)
#    PATH_TO_CKPT = FLAGS.PATH_TO_CKPT
#    GPU_MEMORY = FLAGS.GPU_MEMORY
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=GPU_MEMORY)
    config=tf.ConfigProto(gpu_options=gpu_options)
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
#    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
            int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#    cv2.namedWindow("surveillance")
    frames = 0
    boxes = {}
    pedestrians = {}
    counter = 0
    starttime = time.time()
    with detection_graph.as_default():
        with tf.Session(config=config,graph=detection_graph) as sess:
            success, frame = videoCapture.read()
            while success:
                print (" -------------------- FRAME %d --------------------" % frames)

                if frames % DETECT_FREQUECY == 0:
                    ok ,output_dict_pedestrian,output_dict_helmet = detect_for_single_img(frame, detection_graph, sess,size)
#                    if frames == 40:
#                        break
                    helmet_boxes = []
                    if len(output_dict_helmet) > 0:
                        
                        helmet_boxes = output_dict_helmet['detection_boxes']
                    
                    if len(boxes) is 0 and ok :
                        pedestrian_boxes = output_dict_pedestrian["detection_boxes"]
                        for i,pedestrian_box in enumerate(pedestrian_boxes):
                            boxes[counter] = pedestrian_box
                            pedestrians[counter] = Pedestrian(counter,frame, pedestrian_box,CAMERA_ID)
                            pedestrians[counter].helmet_detection(helmet_boxes)
                            counter += 1
                            
                    elif ok and len(output_dict_pedestrian)>0 and len(output_dict_pedestrian["detection_boxes"]) > 0 :
                        pedestrian_boxes = output_dict_pedestrian["detection_boxes"]
                        assignments, unassigned_tracks, unassigned_detections = detection_to_track_assignment(boxes,pedestrian_boxes)
                        
                        if len(assignments) >0 :
                            for k,v in assignments.items():
                                pedestrian_box = pedestrian_boxes[v]
                                pedestrians[k].correct(frame,pedestrian_box)
                                pedestrians[k].helmet_detection(helmet_boxes)
                                boxes[k] = pedestrian_boxes[v]
                        
                        if len(unassigned_detections) >0 :
                            for i in unassigned_detections:
    #                            print(i)
                                pedestrian_box = pedestrian_boxes[i]
                                boxes[counter] = pedestrian_box
                                pedestrians[counter] = Pedestrian(counter,frame, pedestrian_box,CAMERA_ID)
                                pedestrians[counter].helmet_detection(helmet_boxes)
                                counter += 1
                        if len(unassigned_tracks) > 0 :
                            for i in unassigned_tracks:
                                pedestrians[i].invisible()
                                pedestrians[i].helmet_detection(helmet_boxes)
                            print("unassigned_tracks")
                IDs = []
                for i, p in pedestrians.items():
                    ok,ID,newbox = p.update(frame)
                    boxes[ID] = newbox
                    if ok:
                        IDs.append(ID)
                if len(IDs)>0:
                    for ID in IDs:
                        pedestrians.pop(ID)
                        boxes.pop(ID)
#                cv2.imshow('surveillance', frame) #显示
#                cv2.imwrite('/home/drtian/ZDGJ/tmp/'+ str(frames)+'.jpg',frame)
                frames += 1
                success, frame = videoCapture.read()
                if cv2.waitKey(1) & 0xff == 27:
                    break
    videoCapture.release()
#    cv2.destroyAllWindows()
    endtime = time.time()
    timedur= endtime - starttime
    print(os.getpgid + ' timedur :'+ str(timedur))
    return(True)
#test_main(CAMERA_IP = '/home/drtian/ZDGJ/shipinsucai/41529035350_4391FC76/71DA15898.mp4')
#main()


