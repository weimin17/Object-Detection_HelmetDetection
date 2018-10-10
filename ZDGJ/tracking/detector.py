#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 16:27:39 2018

@author: drtian
"""
import tensorflow as tf
import numpy as np
import cv2
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util





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
    obj_index = [i for i, x in enumerate(output_dict['detection_scores']) if x > 0.9]
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
