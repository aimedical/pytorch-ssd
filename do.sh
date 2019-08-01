#!/bin/bash -e

BASE_NET_PATH=models/mobilenet_v1_with_relu_69_5.pth
BASE_NET_URL=https://storage.googleapis.com/models-hao/mobilenet_v1_with_relu_69_5.pth
ML_ROOT=/mnt/ml

if [[ ! -f $BASE_NET_PATH ]]; then
    wget -p models $BASE_NET_URL
fi

python train_ssd.py \
       --dataset_type coco \
       --datasets $ML_ROOT/data/annotator-media \
       --coco_ann_path $ML_ROOT/data/ec01/001-test/train.json \
       --validation_dataset_root $ML_ROOT/data/annotator-media \
       --validation_coco_ann_path $ML_ROOT/data/ec01/001-test/val.json \
       --net mb1-ssd \
       --base_net $BASE_NET_PATH \
       --batch_size 24 \
       --num_epochs 200 \
       --scheduler cosine \
       --lr 0.01 \
       --t_max 200
