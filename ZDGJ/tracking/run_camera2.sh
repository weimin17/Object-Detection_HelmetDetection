CUDA_VISIBLE_DEVICES=1 python main_test.py \
--CAMERA_ID=2 \
--CAMERA_IP=/data/ZDGJ/demoVideo/CESM小室-2.mp4 \
--PATH_TO_CKPT=/data/saved_models/ssd_mobilenet_v2/savedModel_120000_CEMS_ssdmobilenetv2_0727/frozen_inference_graph.pb \
--DETECT_FREQUECY=50 \
--GPU_MEMORY=0.4


