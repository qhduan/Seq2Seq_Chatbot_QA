#!/usr/bin/env python3

__author__ = 'qhduan@memect.co'

import sys
import math
import time
import random

import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tqdm import tqdm

import data_util

tf.device(data_util.test_device)

encoder_inputs = [tf.placeholder(tf.int32, [None], name='encoder_inputs_{}'.format(i))
                  for i in range(data_util.input_len)]
decoder_inputs = [tf.placeholder(tf.int32, [None], name='decoder_inputs_{}'.format(i))
                  for i in range(data_util.output_len)]
decoder_targets = [tf.placeholder(tf.int32, [None], name='decoder_targets_{}'.format(i))
                   for i in range(data_util.output_len)]
decoder_weights = [tf.placeholder(tf.float32, [None], name='decoder_weights_{}'.format(i))
                   for i in range(data_util.output_len)]

outputs, states = data_util.build_model(encoder_inputs, decoder_inputs, True)

loss_func = tf.nn.seq2seq.sequence_loss(
    outputs,
    decoder_targets,
    decoder_weights,
    data_util.dim
)

sess = tf.Session()

init = tf.initialize_all_variables()

sess.run(init)

data_util.load_model(sess)

def test_sentence(s):
    s = s.strip()
    if len(s) > data_util.input_len:
        s = s[:data_util.input_len]
    encoder, decoder = data_util.get_sentence(s)
    feed_dict = {}
    for i in range(len(encoder_inputs)):
        feed_dict[encoder_inputs[i]] = encoder[i]
    feed_dict[decoder_inputs[0]] = decoder[0]
    output = sess.run(outputs, feed_dict)
    output = np.asarray(output).argmax(axis=2).T
    for o in output:
        return data_util.indice_sentence(o)

def test_qa(s):
    o = test_sentence(s)
    print('Q:', s)
    print(o)
    print('-' * 10)

def test_example():
    t = [
        '你好',
        '你是谁',
        '你从哪来',
        '你到哪去'
    ]
    for x in t:
        test_qa(x)

def test_db():
    asks, answers = data_util.read_db('db/conversation.db')
    for _ in range(20):
        s = random.choice(asks)
        test_qa(s)

if __name__ == '__main__':
    while True:
        sentence = input('说：')
        sentence = sentence.strip()
        if sentence in ('quit', 'exit'):
            break
        if len(sentence) <= 0:
            break
        recall = test_sentence(sentence)
        print(recall)
