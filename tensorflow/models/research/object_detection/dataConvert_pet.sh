python dataset_tools/create_pet_tf_record.py \
    --label_map_path=data/pet_label_map.pbtxt \
    --data_dir=`pwd` \
    --output_dir=/home/drtian/tensorflow/models/research/object_detection/data_pet/ \
    --faces_only=False