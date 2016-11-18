#!/bin/bash


if false
then

echo "##########"

python3 train.py \
--size 512 \
--num_layers 2 \
--num_epoch 5 \
--batch_size 512 \
--num_per_epoch 1000000 \
--mutual_info true \
--mutual_info_lambda 0.0 \
--model_dir ./model/model_bleu_1

python3 train.py \
--size 512 \
--num_layers 2 \
--num_epoch 5 \
--batch_size 512 \
--num_per_epoch 1000000 \
--mutual_info true \
--mutual_info_lambda 0.0 \
--model_dir ./model/model_bleu_1 --bleu 5000
fi

echo "##########"

python3 train.py \
--size 512 \
--num_layers 2 \
--num_epoch 5 \
--batch_size 512 \
--num_per_epoch 1000000 \
--mutual_info true \
--mutual_info_lambda 0.2 \
--model_dir ./model/model_bleu_2

python3 train.py \
--size 512 \
--num_layers 2 \
--num_epoch 5 \
--batch_size 512 \
--num_per_epoch 1000000 \
--mutual_info true \
--mutual_info_lambda 0.2 \
--model_dir ./model/model_bleu_2 --bleu 5000

echo "##########"

python3 train.py \
--size 512 \
--num_layers 2 \
--num_epoch 5 \
--batch_size 512 \
--num_per_epoch 1000000 \
--mutual_info true \
--mutual_info_lambda 0.4 \
--model_dir ./model/model_bleu_3

python3 train.py \
--size 512 \
--num_layers 2 \
--num_epoch 5 \
--batch_size 512 \
--num_per_epoch 1000000 \
--mutual_info true \
--mutual_info_lambda 0.4 \
--model_dir ./model/model_bleu_3 --bleu 5000

echo "##########"

python3 train.py \
--size 512 \
--num_layers 2 \
--num_epoch 5 \
--batch_size 512 \
--num_per_epoch 1000000 \
--mutual_info true \
--mutual_info_lambda 0.6 \
--model_dir ./model/model_bleu_4

python3 train.py \
--size 512 \
--num_layers 2 \
--num_epoch 5 \
--batch_size 512 \
--num_per_epoch 1000000 \
--mutual_info true \
--mutual_info_lambda 0.6 \
--model_dir ./model/model_bleu_4 --bleu 5000

echo "##########"

python3 train.py \
--size 512 \
--num_layers 2 \
--num_epoch 5 \
--batch_size 512 \
--num_per_epoch 1000000 \
--mutual_info true \
--mutual_info_lambda 0.8 \
--model_dir ./model/model_bleu_5

python3 train.py \
--size 512 \
--num_layers 2 \
--num_epoch 5 \
--batch_size 512 \
--num_per_epoch 1000000 \
--mutual_info true \
--mutual_info_lambda 0.8 \
--model_dir ./model/model_bleu_5 --bleu 5000
