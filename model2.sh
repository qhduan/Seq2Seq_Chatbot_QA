#!/bin/bash

if [ -z $1 ]; then
    test="false"
else
    test="true"
fi

python3 train.py \
--size 512 \
--num_layers 2 \
--num_epoch 20 \
--batch_size 512 \
--num_per_epoch 1000000 \
--test $test \
--mutual_info true \
--model_dir ./model/model2
