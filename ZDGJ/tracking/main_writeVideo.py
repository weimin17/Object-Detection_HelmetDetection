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
from Pedestrian import Pedestrian
from detector import detect_for_single_img

from moviepy.editor import VideoFileClip

#tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('CAMERA_ID', '', 'CAMERA_ID')
flags.DEFINE_string('CAMERA_IP', '', 'CAMERA_IP')
flags.DEFINE_string('PATH_TO_CKPT', '', 'PATH_TO_CKPT')
flags.DEFINE_integer('DETECT_FREQUECY', 20, 'DETECT_FREQUECY')
flags.DEFINE_float('GPU_MEMORY', 0.4, 'GPU_MEMORY')

#flags.DEFINE_integer('ALARM_LEVEL', 20, 'ALARM_LEVEL')
#flags.DEFINE_integer('ALARM_PERIOD', 100, 'ALARM_PERIOD')

FLAGS = flags.FLAGS

def main(_):
    
#    videoCapture = cv2.VideoCapture('/home/drtian/ZDGJ/监控录像/1#机组脱硫CESM小室1529636867_2CE8C22B/未命名项目.mp4')
#    PATH_TO_CKPT = '/home/drtian/ZDGJ/FAST_RCNN_ALL_aug_3/frozen_inference_graph.pb'
#    DETECT_FREQUECY = 25 
#    CAMERA_ID = 1
#    GPU_MEMORY = 1
    CAMERA_ID = FLAGS.CAMERA_ID
    DETECT_FREQUECY = FLAGS.DETECT_FREQUECY
    videoCapture = cv2.VideoCapture(FLAGS.CAMERA_IP)
    PATH_TO_CKPT = FLAGS.PATH_TO_CKPT
    GPU_MEMORY = FLAGS.GPU_MEMORY
    
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=GPU_MEMORY)
    config=tf.ConfigProto(gpu_options=gpu_options)
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
            int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, fps, size)
#    cv2.namedWindow("surveillance")
    frames = 0
    boxes = {}
    pedestrians = {}
    counter = 0

    with detection_graph.as_default():
    
        with tf.Session(config=config,graph=detection_graph) as sess:

            while success:
                print (" -------------------- FRAME %d --------------------" % frames)
                success, frame = videoCapture.read()
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
                out.write(frame)
#                cv2.imshow('surveillance', frame) #显示
#                cv2.imwrite('/home/drtian/ZDGJ/tmp/'+ str(frames)+'.jpg',frame)
                frames += 1
                success, frame = videoCapture.read()
#                if cv2.waitKey(1) & 0xff == 27:
#                    break
    videoCapture.release()
    clip = VideoFileClip('/home/drtian/ZDGJ/监控录像/1#机组脱硫CESM小室1529636867_2CE8C22B/fix_16869FE8_1529636867_1.mp4')
    print(clip.duration)
    
#    cv2.destroyAllWindows()

if __name__ == '__main__':
  tf.app.run()






  
  
  
