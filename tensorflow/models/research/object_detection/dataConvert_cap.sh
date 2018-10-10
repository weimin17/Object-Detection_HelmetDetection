
python dataset_tools/create_pet_tf_record.py \
    --label_map_path=/home/drtian/tensorflow/models/research/object_detection/data_safetyHelmet/cap_label_map.pbtxt \
    --data_dir=/home/drtian/co/safecap/train \
    --output_dir=/home/drtian/tensorflow/models/research/object_detection/data_safetyHelmet/ \
    --faces_only=False