from __future__ import division

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 03 13:57:45 2018

@author: weimin
"""

from multiprocessing import Pool
import numpy as np
import sys, os, time
import tensorflow as tf
import cv2, csv
import shutil

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops


from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util
#Disable al debugging logs
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

PATH_TO_LABELS = '/data/contest/models/faster_RCNN_Inception_v2/helmet_label_map.pbtxt'

PATH_TO_CKPT = '/data/contest/savedModels/savedModel_100000_train_faster_RCNNv2_0802/frozen_inference_graph.pb'

NUM_CLASSES = 1
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


def run_inference_for_single_image(image, graph):

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

def detect_singe_img(image_path, write_path, filename):
    print(image_path)
    img = cv2.imread(image_path)
    # 1920*1080 or others
    size = img.shape
    output_dict = run_inference_for_single_image(img, detection_graph)
    img, bndbox_helmet = vis_util.visualize_boxes_and_labels_on_image_array(
        img,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=4,
        min_score_thresh=.93
    )
    cv2.imwrite(os.path.join(write_path, filename), img)
#    print(filename, bndbox_helmet, size)
    return [filename, bndbox_helmet, size]
#    return bndbox_helmet


def detect_file_imgs(PATH_TO_TEST_IMAGES_DIR,PATH_TO_WRITE):
  imagelist = os.listdir(PATH_TO_TEST_IMAGES_DIR)
  if os.path.exists(PATH_TO_WRITE):
    shutil.rmtree(PATH_TO_WRITE)
    os.makedirs(PATH_TO_WRITE)
  else:
    os.makedirs(PATH_TO_WRITE)
  bndbox_all = []
  for filename in imagelist:
      image_path=os.path.join(PATH_TO_TEST_IMAGES_DIR,filename)
      bndbox_helmet, size = detect_singe_img(image_path,PATH_TO_WRITE,filename)

def mycallback(amount_arr):
    # print(x)
  for x in amount_arr:
    print(x)
    # write to dictionary
    bndbox_dic_all[x[0]] = x[1]

    # write to csv
    if len(x[1]) == 0:
        row = [x[0], ""]
        writer.writerow(row)
    else:
        for i in range(len(x[1])):
            row = [x[0], "%s %s %s %s" % (
            int(x[2][1] * x[1][i][0]), int(x[2][0] * x[1][i][1]), int(x[2][1] * x[1][i][2]), int(x[2][0] * x[1][i][3]))]
            writer.writerow(row)

def package(imagelist,PATH_TO_TEST_IMAGES_DIR,PATH_TO_WRITE):
    amount_arr = []

    with detection_graph.as_default():
      with tf.Session() as sess:
        for filename in imagelist:
            image_path=os.path.join(PATH_TO_TEST_IMAGES_DIR,filename)
            amount_arr.append(detect_singe_img(image_path, PATH_TO_WRITE, filename))
    return amount_arr

if __name__ == '__main__':
    e1 = time.time()
    PATH_TO_TEST_IMAGES_DIR = '/data/contest/data/test_test/'
    PATH_TO_WRITE = '/data/contest/data/aaaaaaaa/'
    imagelist = os.listdir(PATH_TO_TEST_IMAGES_DIR)
    if os.path.exists(PATH_TO_WRITE):
        # print('path exists')
        shutil.rmtree(PATH_TO_WRITE)
        os.makedirs(PATH_TO_WRITE)
    else:
        os.makedirs(PATH_TO_WRITE)
    bndbox_dic_all = {}
    # save to csv
    csvFile = open('/data/contest/data/aaaaaaaa.csv', 'w')
    writer = csv.writer(csvFile)
    n_pool = 2
    p = Pool(n_pool)
    for i in range(n_pool):
        p.apply_async(package,args=(imagelist[int((i)*(9000/n_pool)):int((i+1)*(9000/n_pool))],PATH_TO_TEST_IMAGES_DIR,PATH_TO_WRITE,),callback = mycallback)

    print('Waiting for all subprocesses done..')
    p.close()
    p.join()
    print('merge over')
    csvFile.close()
    # save to np
    np.save('/data/contest/data/aaaaaaaa.npy', bndbox_dic_all)
    print('npy saved!')
    print('Time:' + str(time.time() - e1))

'''
if __name__ == '__main__':
  PATH_TO_TEST_IMAGES_DIR='/data/contest/data/test/'
  PATH_TO_WRITE='/data/contest/data/aaaaaaaaaa/'
  imagelist = os.listdir(PATH_TO_TEST_IMAGES_DIR)
  if os.path.exists(PATH_TO_WRITE):
    shutil.rmtree(PATH_TO_WRITE)
    os.makedirs(PATH_TO_WRITE)
  else:
    os.makedirs(PATH_TO_WRITE)
  bndbox_dic_all = {}
  # save to csv
  csvFile = open('/data/contest/data/aaaaaaaaa.csv','w')
  writer = csv.writer(csvFile)
  with detection_graph.as_default():
    with tf.Session() as sess:
      for filename in imagelist:
          image_path=os.path.join(PATH_TO_TEST_IMAGES_DIR,filename)
          bndbox_helmet, size = detect_singe_img(image_path,PATH_TO_WRITE,filename)
          bndbox_dic_all[filename] = bndbox_helmet
          # write to csv
          if len(bndbox_helmet)==0:
              row = [filename,""]
              writer.writerow(row)
          else:
              for i in range(len(bndbox_helmet)):
                  row = [filename,"%s %s %s %s"%(int(bndbox_helmet[i][0]*size[1]),int(bndbox_helmet[i][1]*size[0]),int(bndbox_helmet[i][2]*size[1]),int(bndbox_helmet[i][3]*size[0]))]
                  writer.writerow(row)

  csvFile.close()

  np.save('/data/contest/data/aaaaaaaa.npy',bndbox_dic_all)
  print('saved!')
'''
