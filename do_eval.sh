#!/bin/bash

python eval_model.py \
       --net_type mb1-ssd \
       --trained_model /tmp/mb1-ssd-Epoch-199-Loss-2.961164056293426.pth \
       --dataset ./data \
       --nms_method hard \
       --output_dir ./output \
       --output ./output.json
       
