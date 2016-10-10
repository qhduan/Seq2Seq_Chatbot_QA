#!/usr/bin/env python3

__author__ = 'qhduan@memect.co'

import os
import json
import math
import sqlite3
from collections import OrderedDict
import tensorflow as tf

import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm

EOS = '<eos>'
UNK = '<unk>'
PAD = '<pad>'
GO = '<go>'

def time(s):
    ret = ''
    if s >= 60 * 60:
        h = math.floor(s / (60 * 60))
        ret += '{}h'.format(h)
        s -= h * 60 * 60
    if s >= 60:
        m = math.floor(s / 60)
        ret += '{}m'.format(m)
        s -= m * 60
    if s >= 1:
        s = math.floor(s)
        ret += '{}s'.format(s)
    return ret

def load_dictionary():
    with open('db/dictionary.json', 'r') as fp:
        dictionary = [EOS, UNK, PAD, GO] + json.load(fp)
        index_word = OrderedDict()
        word_index = OrderedDict()
        for index, word in enumerate(dictionary):
            index_word[index] = word
            word_index[word] = index
        dim = len(dictionary)
    return dim, dictionary, index_word, word_index

def load_config():
    with open('config.json', 'r') as fp:
        config = json.load(fp)
    return (
        config['input_len'],
        config['output_len'],
        config['size'],
        config['depth'],
        config['dropout'],
        config['batch_size'],
        config['random_state'],
        config['learning_rate'],
        config['epoch'],
        config['train_device'],
        config['test_device']
    )

def save_model(sess, name='model.ckpt'):
    if not os.path.exists('model'):
        os.makedirs('model')
    saver = tf.train.Saver()
    saver.save(sess, 'model/' + name)

def load_model(sess, name='model.ckpt'):
    saver = tf.train.Saver()
    saver.restore(sess, 'model/' + name)

dim, dictionary, index_word, word_index = load_dictionary()
input_len, output_len, size, depth, dropout, \
batch_size, random_state, learning_rate, epoch, \
train_device, test_device = load_config()

def build_model(encoder_inputs, decoder_inputs, feed_previous=False, size=size, dropout=dropout, depth=depth, dim=dim):
    cell = tf.nn.rnn_cell.BasicLSTMCell(size)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * depth)
    outputs, states = tf.nn.seq2seq.embedding_attention_seq2seq(
        encoder_inputs,
        decoder_inputs,
        cell,
        dim,
        dim,
        embedding_size=size,
        feed_previous=feed_previous
    )
    return outputs, states

def read_db(db_file, input_len=input_len, output_len=output_len-2, tolerate_unk=1):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    asks = []
    answers = []
    def is_valid(s, l):
        if len(s) > l:
            return False
        unk = 0
        for w in s:
            if w not in word_index:
                unk += 1
                if unk > tolerate_unk:
                    return False
        return True
    ret = c.execute('''SELECT ask, answer FROM conversation''')
    for ask, answer in tqdm(ret):
        if is_valid(ask, input_len) and is_valid(answer, output_len):
            asks.append(ask)
            answers.append(answer)
    return shuffle(asks, answers, random_state=0)

def get_sentence(sentence, input_len=input_len):
    encoder_sequence_batch = []
    decoder_sequence_batch = []
    #
    encoder = sentence_indice(sentence)
    # trick from Sequence to Sequence Learning with Neural Networks
    encoder.reverse()
    decoder = [word_index[GO]]
    # left padding input
    encoder_sequence = [word_index[PAD]] * (input_len - len(encoder)) + encoder
    decoder_sequence = decoder + [word_index[PAD]]
    #
    encoder_sequence_batch.append(encoder_sequence)
    decoder_sequence_batch.append(decoder_sequence)
    # 需要经过一个转置
    return (
        np.asarray(encoder_sequence_batch).T,
        np.asarray(decoder_sequence_batch).T
    )

def get_batch(asks, answers, batch_size=batch_size, input_len=input_len, output_len=output_len):
    while True:
        encoder_sequence_batch = []
        decoder_sequence_batch = []
        target_sequence_batch = []
        weights_sequence_batch = []
        for ask, answer in zip(asks, answers):
            if len(encoder_sequence_batch) >= batch_size:
                # 需要经过一个转置
                yield (
                    np.asarray(encoder_sequence_batch).T,
                    np.asarray(decoder_sequence_batch).T,
                    np.asarray(target_sequence_batch).T,
                    np.asarray(weights_sequence_batch).T
                )
                encoder_sequence_batch = []
                decoder_sequence_batch = []
                target_sequence_batch = []
                weights_sequence_batch = []
            encoder = sentence_indice(ask)
            # trick from Sequence to Sequence Learning with Neural Networks
            encoder.reverse()
            target = sentence_indice(answer) + [word_index[EOS]]
            decoder = [word_index[GO]] + target
            # left padding input
            encoder_sequence = [word_index[PAD]] * (input_len - len(encoder)) + encoder
            decoder_sequence = decoder + [word_index[PAD]] * (output_len - len(decoder))
            target_sequence = target + [word_index[PAD]] * (output_len - len(target))
            weights_sequence = [1.0] * len(target) + [0.0] * (output_len - len(target))
            encoder_sequence_batch.append(encoder_sequence)
            decoder_sequence_batch.append(decoder_sequence)
            target_sequence_batch.append(target_sequence)
            weights_sequence_batch.append(weights_sequence)

def sentence_indice(sentence):
    ret = []
    for  word in sentence:
        if word in word_index:
            ret.append(word_index[word])
        else:
            ret.append(word_index[UNK])
    return ret

def indice_sentence(indice):
    ret = []
    for index in indice:
        word = index_word[index]
        if word == EOS:
            break
        if word != UNK and word != GO and word != PAD:
            ret.append(word)
    return ''.join(ret)

def vector_sentence(vector):
    return indice_sentence(vector.argmax(axis=1))


if __name__ == '__main__':
    pass
