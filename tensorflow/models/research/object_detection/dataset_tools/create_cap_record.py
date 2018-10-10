# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert the Oxford pet dataset to TFRecord for object_detection.

See: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar
     Cats and Dogs
     IEEE Conference on Computer Vision and Pattern Recognition, 2012
     http://www.robots.ox.ac.uk/~vgg/data/pets/

Example usage:
    python object_detection/dataset_tools/create_pet_tf_record.py \
        --data_dir=/home/user/pet \
        --output_dir=/home/user/pet/output
"""

import io
import logging
import os
import random

from lxml import etree
import numpy as np
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

#flags = tf.app.flags
#flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
#flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
#flags.DEFINE_string('label_map_path', 'data/pet_label_map.pbtxt',
#                    'Path to label map proto')
#flags.DEFINE_boolean('faces_only', True, 'If True, generates bounding boxes '
#                     'for pet faces.  Otherwise generates bounding boxes (as '
#                     'well as segmentations for full pet bodies).  Note that '
#                     'in the latter case, the resulting files are much larger.')
#flags.DEFINE_string('mask_type', 'png', 'How to represent instance '
#                    'segmentation masks. Options are "png" or "numerical".')
#FLAGS = flags.FLAGS
def dict_to_tf_example(data,label_map_dict,image_dir):
  # TODO(user): Populate the following variables from your example.
  height = int(data['size']['height']) # Image height
  width = int(data['size']['width']) # Image width
  s = data['filename']
  if s[(len(s)-3):len(s)] == 'png':
      s = s[0 :(len(s)-3)]+'jpg' 
  filename = os.path.join(image_dir,s) # Filename of the image. Empty if image is not from file
#  encoded_image_data = None # Encoded image bytes
  with tf.gfile.GFile(filename, 'rb') as fid:
      encoded_image_data = fid.read()
  encoded_image_data_io = io.BytesIO(encoded_image_data)
  image = PIL.Image.open(encoded_image_data_io)
  
  if image.format != 'JPEG':
      raise ValueError('Image format not JPEG')
#  key = hashlib.sha256(encoded_image_data).hexdigest()
  
  image_format = b'jpeg' # b'jpeg' or b'png'

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)
  
  if 'object' in data :
      for obj in data['object']:
          xmin = float(obj['bndbox']['xmin'])
          xmax = float(obj['bndbox']['xmax'])
          ymin = float(obj['bndbox']['ymin'])
          ymax = float(obj['bndbox']['ymax'])
          xmins.append(xmin / width)
          ymins.append(ymin / height)
          xmaxs.append(xmax / width)
          ymaxs.append(ymax / height)
          class_name = obj['name']
          classes_text.append(class_name.encode('utf8'))
          classes.append(label_map_dict[class_name])



  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example

def create_tf_record(output_filename,label_map_dict,annotations_dir,image_dir,examples):
#    output_filename = train_output_path
#    examples = train_examples
    writer = tf.python_io.TFRecordWriter(output_filename)
    
    for idx, example in enumerate(examples):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(examples))
        xml_path = os.path.join(annotations_dir, example + '.xml')
        if not os.path.exists(xml_path):
            logging.warning('Could not find %s, ignoring example.', xml_path)
            continue
        with tf.gfile.GFile(xml_path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        try:
            tf_example = dict_to_tf_example(
                    data,
                    label_map_dict,
                    image_dir)
            writer.write(tf_example.SerializeToString())
        except ValueError:
            logging.warning('Invalid example: %s, ignoring.', xml_path)

    writer.close()






# TODO(derekjchow): Add test for pet/PASCAL main files.
def main(_):
#  data_dir = FLAGS.data_dir
  data_dir = '/home/drtian/co/safecap/train'
#  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
  label_map_dict = label_map_util.get_label_map_dict('/home/drtian/tensorflow/models/research/object_detection/data_safetyHelmet/cap_label_map.pbtxt')
  logging.info('Reading from Pet dataset.')
  image_dir = os.path.join(data_dir, 'JPEGImages')
  annotations_dir = os.path.join(data_dir, 'Annotations')
#  examples_path = os.path.join(annotations_dir, 'trainval.txt')
#  examples_list = dataset_util.read_examples_list(examples_path)
  examples_list = os.listdir(image_dir)
  
  examples_list = list(map(lambda s:s[0:(len(s)-4)],examples_list))
 
  
  
  # Test images are not included in the downloaded data set, so we shall perform
  # our own split.
  random.seed(42)
  random.shuffle(examples_list)
  num_examples = len(examples_list)
  num_train = int(0.7 * num_examples)
  train_examples = examples_list[:num_train]
  val_examples = examples_list[num_train:]
  logging.info('%d training and %d validation examples.',
               len(train_examples), len(val_examples))
#
#  train_output_path = os.path.join(FLAGS.output_dir, 'pet_train.record')
#  val_output_path = os.path.join(FLAGS.output_dir, 'pet_val.record')
  train_output_path = os.path.join('/home/drtian/tensorflow/models/research/object_detection/data_safetyHelmet', 'cap_train.record')
  val_output_path = os.path.join('/home/drtian/tensorflow/models/research/object_detection/data_safetyHelmet', 'cap_val.record')


  create_tf_record(
      train_output_path,
      label_map_dict,
      annotations_dir,
      image_dir,
      train_examples
      )
  
  create_tf_record(
      val_output_path,
      label_map_dict,
      annotations_dir,
      image_dir,
      val_examples
      )


if __name__ == '__main__':
  tf.app.run()
