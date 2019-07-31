#!/bin/bash

python train_ssd.py \
       --dataset_type coco \
       --datasets /ml-root/data/annotator-media \
       --coco_ann_path /ml-root/data/ec01/001-test/train.json \
       --validation_dataset ./ssd \
       --net mb1-ssd \
       --base_net models/mobilenet_v1_with_relu_69_5.pth \
       --batch_size 24 \
       --num_epochs 200 \
       --scheduler cosine \
       --lr 0.01 \
       --t_max 200
