#!/usr/bin/env python3

__author__ = 'qhduan@memect.co'

import os
import json
import math
import pickle
import sqlite3
from collections import OrderedDict

import numpy as np
from tqdm import tqdm
import tensorflow as tf
from sklearn.utils import shuffle

def with_path(p):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, p)

DICTIONARY_PATH = 'db/dictionary.json'
EOS = '<eos>'
UNK = '<unk>'
PAD = '<pad>'
GO = '<go>'

buckets = [
    (5, 15),
    (10, 20),
    (15, 25),
    (20, 30)
]

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
    with open(with_path(DICTIONARY_PATH), 'r') as fp:
        dictionary = [EOS, UNK, PAD, GO] + json.load(fp)
        index_word = OrderedDict()
        word_index = OrderedDict()
        for index, word in enumerate(dictionary):
            index_word[index] = word
            word_index[word] = index
        dim = len(dictionary)
    return dim, dictionary, index_word, word_index

def save_model(sess, name='model.ckpt'):
    if not os.path.exists('model'):
        os.makedirs('model')
    saver = tf.train.Saver()
    saver.save(sess, with_path('model/' + name))

def load_model(sess, name='model.ckpt'):
    saver = tf.train.Saver()
    saver.restore(sess, with_path('model/' + name))

dim, dictionary, index_word, word_index = load_dictionary()

print('dim: ', dim)

EOS_ID = word_index[EOS]
UNK_ID = word_index[UNK]
PAD_ID = word_index[PAD]
GO_ID = word_index[GO]

class BucketData(object):

    def __init__(self, buckets_dir, encoder_size, decoder_size):
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.name = 'bucket_%d_%d.db' % (encoder_size, decoder_size)
        self.path = os.path.join(buckets_dir, self.name)
        self.conn = sqlite3.connect(self.path)
        self.cur = self.conn.cursor()
        sql = '''SELECT MAX(ROWID) FROM conversation;'''
        self.size = self.cur.execute(sql).fetchall()[0][0]

    def random(self):
        while True:
            # 选择一个[1, MAX(ROWID)]中的整数，读取这一行
            rowid = np.random.randint(1, self.size + 1)
            sql = '''
            SELECT ask, answer FROM conversation
            WHERE ROWID = {};
            '''.format(rowid)
            ret = self.cur.execute(sql).fetchall()
            if len(ret) == 1:
                ask, answer = ret[0]
                if ask is not None and answer is not None:
                    return ask, answer

def read_bucket_dbs(buckets_dir):
    ret = []
    for encoder_size, decoder_size in buckets:
        bucket_data = BucketData(buckets_dir, encoder_size, decoder_size)
        ret.append(bucket_data)
    return ret

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

def generate_bucket_dbs(
        input_dir,
        output_dir,
        buckets,
        tolerate_unk=1
    ):
    pool = {}
    def _get_conn(key):
        if key not in pool:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            name = 'bucket_%d_%d.db' % key
            path = os.path.join(output_dir, name)
            conn = sqlite3.connect(path)
            cur = conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS conversation (ask text, answer text);""")
            conn.commit()
            pool[key] = (conn, cur)
        return pool[key]
    all_inserted = {}
    for encoder_size, decoder_size in buckets:
        key = (encoder_size, decoder_size)
        all_inserted[key] = 0
    # 从input_dir列出数据库列表
    db_paths = []
    for dirpath, _, filenames in os.walk(input_dir):
        for filename in (x for x in sorted(filenames) if x.endswith('.db')):
            db_path = os.path.join(dirpath, filename)
            db_paths.append(db_path)
    # 对数据库列表中的数据库挨个提取
    for db_path in db_paths:
        print('读取数据库: {}'.format(db_path))
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        def is_valid(s):
            unk = 0
            for w in s:
                if w not in word_index:
                    unk += 1
                    if unk > tolerate_unk:
                        return False
            return True
        # 读取最大的rowid，如果rowid是连续的，结果就是里面的数据条数
        # 比SELECT COUNT(1)要快
        total = c.execute('''SELECT MAX(ROWID) FROM conversation;''').fetchall()[0][0]
        ret = c.execute('''SELECT ask, answer FROM conversation;''')
        wait_insert = []
        def _insert(wait_insert):
            if len(wait_insert) > 0:
                for encoder_size, decoder_size, ask, answer in wait_insert:
                    key = (encoder_size, decoder_size)
                    conn, cur = _get_conn(key)
                    cur.execute("""
                    INSERT INTO conversation (ask, answer) VALUES ('{}', '{}');
                    """.format(ask.replace("'", "''"), answer.replace("'", "''")))
                    all_inserted[key] += 1
                for conn, _ in pool.values():
                    conn.commit()
                wait_insert = []
            return wait_insert
        for ask, answer in tqdm(ret, total=total):
            if is_valid(ask) and is_valid(answer):
                for i in range(len(buckets)):
                    encoder_size, decoder_size = buckets[i]
                    if len(ask) <= encoder_size and len(answer) < decoder_size:
                        wait_insert.append((encoder_size, decoder_size, ask, answer))
                        if len(wait_insert) > 1000000:
                            wait_insert = _insert(wait_insert)
                        break
    wait_insert = _insert(wait_insert)
    return all_inserted

if __name__ == '__main__':
    print('generate bucket dbs')
    all_inserted = generate_bucket_dbs('./db', './bucket_dbs', buckets, 1)
    for key, inserted_count in all_inserted.items():
        print(key)
        print(inserted_count)
    print('done')
