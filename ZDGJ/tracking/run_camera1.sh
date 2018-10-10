CUDA_VISIBLE_DEVICES=1 python main_test.py \
--CAMERA_ID=1 \
--CAMERA_IP=/data/ZDGJ/demoVideo/平电中路-1.mp4 \
--PATH_TO_CKPT=/data/FAST_RCNN_ALL_aug_3/frozen_inference_graph.pb \
--DETECT_FREQUECY=25 \
--GPU_MEMORY=0.04

#--PATH_TO_CKPT=/data/saved_models/ssd_mobilenet_v2/savedModel_200000_turbine_ssdmobilenetv2_0725/frozen_inference_graph.pb
