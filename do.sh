#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python train_ssd.py \
       --dataset_type coco \
       --datasets ./ssd \
       --validation_dataset ./ssd \
       --net mb1-ssd \
       --base_net models/mobilenet_v1_with_relu_69_5.pth \
       --batch_size 24 \
       --num_epochs 200 \
       --scheduler cosine \
       --lr 0.01 \
       --t_max 200
