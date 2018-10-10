for video_single in '/data/video/speed_test/fivePeople_20s_1920_1080.mp4' '/data/video/speed_test/fivePeople_20s_1280_720.mp4' '/data/video/speed_test/onePeople_20s_1920_1080.mp4' '/data/video/speed_test/onePeople_20s_1280_720.mp4' '/data/video/speed_test/noPeople_20s_1920_1080.mp4' '/data/video/speed_test/noPeople_20s_1280_720.mp4'
do
  CUDA_VISIBLE_DEVICES=0 python main.py \
  --CAMERA_ID=5 \
  --CAMERA_IP=$video_single \
  --PATH_TO_CKPT=/data/FAST_RCNN_ALL_aug_3/frozen_inference_graph.pb \
  --DETECT_FREQUECY=25 \
  --GPU_MEMORY=0.6
done
#  --PATH_TO_CKPT=/data/saved_models/ssd_mobilenet_v2/savedModel_100000_all_ssdmobilenetv2_0727/frozen_inference_graph.pb \
