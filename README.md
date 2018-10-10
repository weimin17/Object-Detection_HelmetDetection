# Tensorflow Object Detection - HelmetDetection

This repository shows how to deploy Object Detection API - Tensorflow and give specific tutorials. It is based on [Object Detection](https://github.com/tensorflow/models/tree/master/research/object_detection). A specific example is to detect helmet, it could also explore to any other object detection tasks.

## Installation

* [Installation](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

## Data Preparation

*Step 1*. Label - Generate the XML files based on original images, which means you locate the object in images and record the information into XML files. Label the image using [lablelImg](https://github.com/tzutalin/labelImg). The detailed explanation can be found [here](https://www.youtube.com/watch?v=K_mFnvzyLvc&list=PLQVvvaa0QuDcNK5GeCQnxYnSSaar2tpku&index=3). The Training data needs to follow the organization:

```
--data/train
    --Annotations
    --JPEGImages
```

If you have already have the datasets containing XML and JPG files, then you can skip to Step 2.

*Step 2*. Generate TF-Records. TensorFlow object detection API doesn’t take XML or Image files as an input, but it needs record files to train the model. In the example, I used `/data/cap_train.record` and `/data/cap_val.record` as input. You can generate the files using `/codes/create_helmet_record.py`, just remember to change the dir:

```
#  data_dir = FLAGS.data_dir
data_dir = '../data/train/'
#  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
label_map_dict = label_map_util.get_label_map_dict('../models/faster_RCNN_Inception_v2/helmet_label_map.pbtxt')
```

`/models/faster_RCNN_Inception_resnet_v2/helmet_label_map.pbtxt` is a file for labeling. Give label name i.e helmet in my example. If in case you have multiple labels, increase id number starting from 1 and give appropriate label name.

```
item{
	id: 1
	name: 'helmet'
}
```

```
  num_train = int(0.8 * num_examples)
  train_examples = examples_list[:num_train]
  val_examples = examples_list[num_train:]
```

Above codes means how many datasets you split as train data and test data. I used `0.8` here.

## Train Model
Once the records files are ready, it's ready to train the model.

1.Choose the pre-trained model to be used. You could download [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md), which contains various pre-trained models. My example uses `/models/faster_RCNN_Inception_resnet_v2/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28` for a better accuracy while with slower speed, and you could exchange to anyother models.

2.Remember to use the consistent config file for the same model. My example uses `/models/faster_RCNN_Inception_resnet_v2/faster_rcnn_inception_resnet_v2_atrous_coco_alldata.config`. Remember to change file dir in config file. Remember to change the number of classes in the file according to specific requirement. [Configre Document](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md), [TensorFlow Object Detection API tutorial](https://becominghuman.ai/tensorflow-object-detection-api-tutorial-training-and-evaluating-custom-object-detector-ed2594afcf73) shows more details on how to set parameters.

3.In `/codes/` folder, execute `bash train.sh`, remember to change the dir. You can also execute the command in Terminal. The training process requires tons of time. If you use GPU, it will occupy all of the GPU and you could use the command `CUDA_VISIBLE_DEVICES=0 bash train.sh` to assign a specific GPU.

```
# Train model
python3 ../tensorflow/models/research/train.py \
    --logtostderr \
    --train_dir=../output/output_100000_train_faster_RCNN_Inception_ResNet_v2_0802 \
    --pipeline_config_path=../models/faster_RCNN_Inception_resnet_v2/faster_rcnn_inception_resnet_v2_atrous_coco_alldata.config
```

## Save Model

Before Evaluating, models must be saved. The codes is in `/codes/train.sh`.

```
# save Model
python ../tensorflow/models/research/object_detection/export_inference_graph.py \
--input_type=image_tensor \
--pipeline_config_path=../models/faster_RCNN_Inception_resnet_v2/faster_rcnn_inception_resnet_v2_atrous_coco_alldata.config \
--trained_checkpoint_prefix=../output/output_100000_train_faster_RCNN_Inception_ResNet_v2_0802/model.ckpt \
--output_directory=../savedModels/savedModel_100000_train_faster_RCNN_Inception_ResNet_v2_0802
```

## Evaluate Model

The final step is to evaluate the trained model saved in `../output/` directory. There are two ways to evaluate, using *eval.py* to calculating the total accuracy, or using *object_detection_pic.py* to evaluate a single image. The following shows both two ways.

### Calculating MAP@n 
Use *eval.py* file and evaluate using following command:

```
python ../tensorflow/models/research/object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=../models/faster_RCNN_Inception_resnet_v2/faster_rcnn_inception_resnet_v2_atrous_coco_alldata.config \
    --checkpoint_dir=../output/output_100000_train_faster_RCNN_Inception_ResNet_v2_0802 \
    --eval_dir=../output/output_100000_train_faster_RCNN_Inception_ResNet_v2_0802/eval/
```

This will save the eval results in *eval/* directory. There is a excellent tools: TensorBoard, which could visualize both training and evaling process. To launch TensorBoard (while you visualize the eval results, it would also visualize the training results at the same time), using following command in a new Terminal and locate the terminal in the file:

```
#To visualize the eval results
tensorboard --logdir=../output/output_100000_train_faster_RCNN_Inception_ResNet_v2_0802/eval/
#TO visualize the training results
tensorboard --logdir=../output/output_100000_train_faster_RCNN_Inception_ResNet_v2_0802
```

Instructions for TensorBoard for the beginners: [TensorBoard](https://github.com/tensorflow/tensorboard).

### Evaluating a single image

The codes are in `/codes/train.sh`, remember to change the dir.

```
# Evaluate Model
python ../codes/object_detection_pic_0.75_train-test_faster_RCNN_Inception_ResNet_v2.py
```


