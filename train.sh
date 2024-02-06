#!/bin/bash

python train.py --dataset_name='JSTL_large_4' \
                --gpu_ids='0,1,2,3' \
                 --optimizer='adam' \
                 --start_faith_epoch=200 \
                 --start_eval_epoch=350 \
                 --lr=5e-5 \
                 --base_mae='60,8,85,60' \
                 --lr_decay_iters=300 \
                 --cls_w=1 \
                 --cls_num=4 \
                 --name='Hrnet_IsBN_again' \
                 --net_name='MDKNet' \
                 --reslt_beta=0.9 \
                 --weight_with_target=False \
                 --model_ema=1 \
                 --model_ema_decay=0.99 \
                 --final_conf=0.5 \
                 --Temp=10 \
                 --print_step=1 \
                 --conf_window=5 \
                 --batch_size=32 \
                 --nThreads=16 \
                 --max_epoch=500 \
                 --eval_per_epoch=1
