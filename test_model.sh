#!/bin/bash

python3 train.py \
--size 256 \
--num_layers 2 \
--num_epoch 5 \
--batch_size 512 \
--num_per_epoch 100000 \
--model_dir ./model/model1 \
--test true
