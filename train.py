#!/usr/bin/env python3

__author__ = 'qhduan@memect.co'

import os
import sys
import math
import json
import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

import data_util

def main():

    tf.set_random_seed(data_util.random_state)
    sess = tf.Session()
    tf.device(data_util.train_device)

    encoder_inputs = [tf.placeholder(tf.int32, [None], name='encoder_inputs_{}'.format(i))
                      for i in range(data_util.input_len)]
    decoder_inputs = [tf.placeholder(tf.int32, [None], name='decoder_inputs_{}'.format(i))
                      for i in range(data_util.output_len)]
    decoder_targets = [tf.placeholder(tf.int32, [None], name='decoder_targets_{}'.format(i))
                       for i in range(data_util.output_len)]
    decoder_weights = [tf.placeholder(tf.float32, [None], name='decoder_weights_{}'.format(i))
                       for i in range(data_util.output_len)]

    print('build model')
    outputs, states = data_util.build_model(encoder_inputs, decoder_inputs)

    loss_func = tf.nn.seq2seq.sequence_loss(
        outputs,
        decoder_targets,
        decoder_weights,
        data_util.dim
    )

    opt = tf.train.AdamOptimizer(
        learning_rate=data_util.learning_rate
    )

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss_func, tvars), 5)

    optimizer = opt.apply_gradients(zip(grads, tvars))

    opt_op = opt.minimize(loss_func)

    init = tf.initialize_all_variables()

    sess.run(init)

    ops = (opt_op, outputs, loss_func)

    print('load data')
    asks, answers = data_util.read_db('db/conversation.db')
    print('size of asks and answers: {}, {}'.format(len(asks), len(answers)))

    print('start train')
    train(
        data_util.epoch, asks, answers,
        encoder_inputs, decoder_inputs, decoder_targets, decoder_weights,
        sess, ops
    )

    print('save model')
    data_util.save_model(sess)

    print('train done')

def train(epoch, asks, answers, encoder_inputs, decoder_inputs, decoder_targets, decoder_weights, sess, ops):
    samples_per_epoch = math.ceil(len(asks) / data_util.batch_size) * data_util.batch_size
    print('samples_per_epoch', samples_per_epoch)

    metrics = '   '.join([
        '\r[{}]',
        '{:.1f}%',
        '{}/{}',
        'accu={:.3f}',
        'loss={:.3f}',
        '{}/{}'
    ])
    for epoch_index in range(1, epoch + 1):
        print('Epoch {}:'.format(epoch_index))
        time_start = time.time()
        epoch_trained = 0
        right_ratio = []
        bars_max = 20
        batch_loss = []
        batch_generator = data_util.get_batch(asks, answers)
        for encoder, decoder, target, weights in batch_generator:
            feed_dict = {}
            for i in range(len(encoder_inputs)):
                feed_dict[encoder_inputs[i]] = encoder[i]
            for i in range(len(decoder_inputs)):
                feed_dict[decoder_inputs[i]] = decoder[i]
                feed_dict[decoder_targets[i]] = target[i]
                feed_dict[decoder_weights[i]] = weights[i]
            _, output, loss = sess.run(ops, feed_dict)
            right = (target == np.asarray(output).argmax(axis=2))
            right = right * weights
            right = np.sum(right) / np.sum(weights)
            right_ratio.append(right)
            epoch_trained += data_util.batch_size
            batch_loss.append(loss)
            time_now = time.time()
            time_spend = time_now - time_start
            time_estimate = time_spend / (epoch_trained / samples_per_epoch)
            percent = min(100, epoch_trained / samples_per_epoch) * 100
            bars = math.floor(percent / 100 * bars_max)
            sys.stdout.write(metrics.format(
                '=' * bars + '-' * (bars_max - bars),
                percent,
                epoch_trained, samples_per_epoch,
                np.mean(right_ratio),
                np.mean(batch_loss),
                data_util.time(time_spend), data_util.time(time_estimate)
            ))
            sys.stdout.flush()
            if epoch_trained >= samples_per_epoch:
                break
        print('\n')

if __name__ == '__main__':
    main()
