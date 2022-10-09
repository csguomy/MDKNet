#!/bin/bash

python3 test.py --dataset_name='JSTL_large_4' \
                --gpu_ids='0,1' \
                --optimizer='adam' \
                --start_eval_epoch=200 \
                --lr=5e-5 \
                --cls_w=1 \
                --base_mae='60,10,100,100' \
                --name='MDKNet_models' \
                --net_name='MDKNet' \
                --weight_with_target=True \
                --final_conf=0.5 \
                --batch_size=2 \
                --nThreads=2 \
                --max_epoch=400 \
                --eval_per_epoch=1

