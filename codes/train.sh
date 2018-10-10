# Train model
python3 ../tensorflow/models/research/train.py \
    --logtostderr \
    --train_dir=../output/output_100000_train_faster_RCNN_Inception_ResNet_v2_0802 \
    --pipeline_config_path=../models/faster_RCNN_Inception_resnet_v2/faster_rcnn_inception_resnet_v2_atrous_coco_alldata.config

# save Model
python ../tensorflow/models/research/object_detection/export_inference_graph.py \
--input_type=image_tensor \
--pipeline_config_path=../models/faster_RCNN_Inception_resnet_v2/faster_rcnn_inception_resnet_v2_atrous_coco_alldata.config \
--trained_checkpoint_prefix=../output/output_100000_train_faster_RCNN_Inception_ResNet_v2_0802/model.ckpt \
--output_directory=../savedModels/savedModel_100000_train_faster_RCNN_Inception_ResNet_v2_0802


# Evaluate Model
python ../codes/object_detection_pic_0.75_train-test_faster_RCNN_Inception_ResNet_v2.py
