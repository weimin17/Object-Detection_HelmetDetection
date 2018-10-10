#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 13:57:45 2018

@author: drtian
"""

import numpy as np
import os
import sys
import tensorflow as tf
import cv2


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from hungary import detection_to_track_assignment
from Pedestrian import Pedestrian


PATH_TO_LABELS = '/home/drtian/co/helmet/data/helmet_label_map.pbtxt'

PATH_TO_CKPT = '/home/drtian/ZDGJ/data/data/savedModel_400000_0702/frozen_inference_graph.pb'

NUM_CLASSES = 2
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph,sess):
  # Get handles to input and output tensors
  ops = tf.get_default_graph().get_operations()
  all_tensor_names = {output.name for op in ops for output in op.outputs}
  tensor_dict = {}
  for key in [
      'num_detections', 'detection_boxes', 'detection_scores',
      'detection_classes', 'detection_masks'
  ]:
    tensor_name = key + ':0'
    if tensor_name in all_tensor_names:
      tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
          tensor_name)
  if 'detection_masks' in tensor_dict:
    # The following processing is only for single image
    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
        detection_masks, detection_boxes, image.shape[0], image.shape[1])
    detection_masks_reframed = tf.cast(
        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
    # Follow the convention by adding back the batch dimension
    tensor_dict['detection_masks'] = tf.expand_dims(
        detection_masks_reframed, 0)
  image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

  # Run inference
  output_dict = sess.run(tensor_dict,
                         feed_dict={image_tensor: np.expand_dims(image, 0)})

  # all outputs are float32 numpy arrays, so convert types as appropriate
  output_dict['num_detections'] = int(output_dict['num_detections'][0])
  output_dict['detection_classes'] = output_dict[
      'detection_classes'][0].astype(np.uint8)
  output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
  output_dict['detection_scores'] = output_dict['detection_scores'][0]
  if 'detection_masks' in output_dict:
    output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict



def detect_for_single_img(frame, detection_graph, sess,size):
    output_dict = run_inference_for_single_image(frame, detection_graph, sess)
    obj_index = [i for i, x in enumerate(output_dict['detection_scores']) if x > 0.1]
    output_dict_pedestrian = dict()
    output_dict_helmet = dict()
    ok = False
    if len(obj_index) > 0 :
        ok = True
        for i in obj_index:
            y,x,h,w = output_dict['detection_boxes'][i]
            y,x,h,w = int(y*size[1]),int(x*size[0]),int(h*size[1]),int(w*size[0])
            cv2.rectangle(frame, (x,y), (w, h), (0, 255, 255), 1)

        for k in output_dict:
            if k == 'num_detections':
                continue
            output_dict[k] = output_dict[k][obj_index]
            
        ## TODO : multi class            
        obj_index_pedestrian  = [i for i, x in enumerate(output_dict['detection_classes']) if x == 1]
        obj_index_helmet =  [i for i, x in enumerate(output_dict['detection_classes']) if x == 2]

        for k in output_dict:
#            print(k)
            if k == 'num_detections':
                continue
            output_dict_pedestrian[k] = output_dict[k][obj_index_pedestrian]
            output_dict_helmet[k] = output_dict[k][obj_index_helmet]
        
        for i,bbox in enumerate(output_dict_pedestrian['detection_boxes']) :
            output_dict_pedestrian['detection_boxes'][i] = int(bbox[0]*size[1]),int(bbox[1]*size[0]),int(bbox[2]*size[1]),int(bbox[3]*size[0])
        for i,bbox in enumerate(output_dict_helmet['detection_boxes']) :
            output_dict_helmet['detection_boxes'][i] = int(bbox[0]*size[1]),int(bbox[1]*size[0]),int(bbox[2]*size[1]),int(bbox[3]*size[0])           
    
    return ok,output_dict_pedestrian,output_dict_helmet



videoCapture = cv2.VideoCapture('/media/drtian/CDROM/监控录像/1#机组脱硫CESM小室1529636867_2CE8C22B/未命名项目.mp4')
# TODO:  camera NO ,model , camera IP 

fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
  
cv2.namedWindow("surveillance")
frames = 0

firstFrame = True 
boxes = {}
pedestrians = {}
counter = 0
with detection_graph.as_default():

    with tf.Session(graph=detection_graph) as sess:
        success, frame = videoCapture.read()
        while success:
            print (" -------------------- FRAME %d --------------------" % frames)
#            cv2.imwrite('/home/drtian/ZDGJ/tmp/'+str(frames)+'detection.jpg',frame)
            success, frame = videoCapture.read()
            if frames % 20 == 0:
                ok ,output_dict_pedestrian,output_dict_helmet = detect_for_single_img(frame, detection_graph, sess,size)
#                if len(output_dict_helmet) >0 :
#                    break

                if len(boxes) is 0 and ok :
                    
                    pedestrian_boxes = output_dict_pedestrian["detection_boxes"]
                    hemet_boxes = output_dict_helmet["detection_boxes"]
                    
                    for i,pedestrian_box in enumerate(pedestrian_boxes):
                        boxes[counter] = pedestrian_box
                        ymin,xmin,ymax,xmax =pedestrian_box
                        bbox1 = (xmin,ymin,xmax-xmin,ymax-ymin)
                        pedestrians[counter] = Pedestrian(counter,frame, bbox1)
                        counter += 1
                
                elif ok and len(output_dict_pedestrian)>0 and len(output_dict_pedestrian["detection_boxes"]) > 0 :
                    pedestrian_boxes = output_dict_pedestrian["detection_boxes"]
                    assignments, unassigned_tracks, unassigned_detections = detection_to_track_assignment(boxes,pedestrian_boxes)
                    if len(assignments) >0 :
                        for k,v in assignments.items():
                            ymin,xmin,ymax,xmax = pedestrian_boxes[v]
                            bbox1 = (xmin,ymin,xmax-xmin,ymax-ymin)
                            pedestrians[k].correct(frame,bbox1)
                            boxes[k] = pedestrian_boxes[v]

                    if len(unassigned_detections) >0 :
                        for i in unassigned_detections:
                            print(i)
                            pedestrian_boxe = pedestrian_boxes[i]
                            boxes[counter] = pedestrian_box
                            ymin,xmin,ymax,xmax =pedestrian_box
                            bbox1 = (xmin,ymin,xmax-xmin,ymax-ymin)
                            pedestrians[counter] = Pedestrian(counter,frame, bbox1)

                            counter += 1
                    if len(unassigned_tracks) >0 :
                        for i in unassigned_tracks:
                            ok,invisible_count = pedestrians[i].invisible(frame)
                            if ok:
                                pedestrians.pop(i) 
                                boxes.pop(i)
                        print("unassigned_tracks")
            IDs = []
            for i, p in pedestrians.items():
                ok,ID = p.update(frame)
                if not ID is None:
                    IDs.append(ID)
            if len(IDs)>0:
                for ID in IDs:
                    pedestrians.pop(ID)
                    boxes.pop(ID)
            cv2.imshow('surveillance', frame) #显示
            cv2.imwrite('/home/drtian/ZDGJ/tmp/'+str(frames)+'.jpg',frame)
            frames += 1
            if cv2.waitKey(1) & 0xff == 27:
                break
videoCapture.release()
cv2.destroyAllWindows()








  
  
  
