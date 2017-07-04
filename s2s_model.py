
import pdb
import random
import copy

import numpy as np
import tensorflow as tf

import data_utils

class S2SModel(object):
    def __init__(self,
                source_vocab_size,
                target_vocab_size,
                buckets,
                size,
                dropout,
                num_layers,
                max_gradient_norm,
                batch_size,
                learning_rate,
                num_samples,
                forward_only=False,
                dtype=tf.float32):
        # init member variales
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # LSTM cells
        cell = tf.contrib.rnn.BasicLSTMCell(size)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
        cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)

        output_projection = None
        softmax_loss_function = None

        # 如果vocabulary太大，我们还是按照vocabulary来sample的话，内存会爆
        if num_samples > 0 and num_samples < self.target_vocab_size:
            print('开启投影：{}'.format(num_samples))
            w_t = tf.get_variable(
                "proj_w",
                [self.target_vocab_size, size],
                dtype=dtype
            )
            w = tf.transpose(w_t)
            b = tf.get_variable(
                "proj_b",
                [self.target_vocab_size],
                dtype=dtype
            )
            output_projection = (w, b)

            def sampled_loss(labels, logits):
                labels = tf.reshape(labels, [-1, 1])
                # 因为选项有选fp16的训练，这里同意转换为fp32
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(logits, tf.float32)
                return tf.cast(
                    tf.nn.sampled_softmax_loss(
                        weights=local_w_t,
                        biases=local_b,
                        labels=labels,
                        inputs=local_inputs,
                        num_sampled=num_samples,
                        num_classes=self.target_vocab_size
                    ),
                    dtype
                )
            softmax_loss_function = sampled_loss

        # seq2seq_f
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            tmp_cell = copy.deepcopy(cell)
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                encoder_inputs,
                decoder_inputs,
                tmp_cell,
                num_encoder_symbols=source_vocab_size,
                num_decoder_symbols=target_vocab_size,
                embedding_size=size,
                output_projection=output_projection,
                feed_previous=do_decode,
                dtype=dtype
            )

        # inputs
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.decoder_weights = []

        # buckets中的最后一个是最大的（即第“-1”个）
        for i in range(buckets[-1][0]):
            self.encoder_inputs.append(tf.placeholder(
                tf.int32,
                shape=[None],
                name='encoder_input_{}'.format(i)
            ))
        # 输出比输入大 1，这是为了保证下面的targets可以向左shift 1位
        for i in range(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(
                tf.int32,
                shape=[None],
                name='decoder_input_{}'.format(i)
            ))
            self.decoder_weights.append(tf.placeholder(
                dtype,
                shape=[None],
                name='decoder_weight_{}'.format(i)
            ))

        targets = [
            self.decoder_inputs[i + 1] for i in range(buckets[-1][1])
        ]

        if forward_only:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs,
                self.decoder_inputs,
                targets,
                self.decoder_weights,
                buckets,
                lambda x, y: seq2seq_f(x, y, True),
                softmax_loss_function=softmax_loss_function
            )
            if output_projection is not None:
                for b in range(len(buckets)):
                    self.outputs[b] = [
                        tf.matmul(
                            output,
                            output_projection[0]
                        ) + output_projection[1]
                        for output in self.outputs[b]
                    ]
        else:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs,
                self.decoder_inputs,
                targets,
                self.decoder_weights,
                buckets,
                lambda x, y: seq2seq_f(x, y, False),
                softmax_loss_function=softmax_loss_function
            )

        params = tf.trainable_variables()
        opt = tf.train.AdamOptimizer(
            learning_rate=learning_rate
        )

        if not forward_only:
            self.gradient_norms = []
            self.updates = []
            for output, loss in zip(self.outputs, self.losses):
                gradients = tf.gradients(loss, params)
                clipped_gradients, norm = tf.clip_by_global_norm(
                    gradients,
                    max_gradient_norm
                )
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                    zip(clipped_gradients, params)
                ))
        # self.saver = tf.train.Saver(tf.all_variables())
        self.saver = tf.train.Saver(
            tf.all_variables(),
            write_version=tf.train.SaverDef.V2
        )

    def step(
        self,
        session,
        encoder_inputs,
        decoder_inputs,
        decoder_weights,
        bucket_id,
        forward_only
    ):
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError(
                "Encoder length must be equal to the one in bucket,"
                " %d != %d." % (len(encoder_inputs), encoder_size)
            )
        if len(decoder_inputs) != decoder_size:
            raise ValueError(
                "Decoder length must be equal to the one in bucket,"
                " %d != %d." % (len(decoder_inputs), decoder_size)
            )
        if len(decoder_weights) != decoder_size:
            raise ValueError(
                "Weights length must be equal to the one in bucket,"
                " %d != %d." % (len(decoder_weights), decoder_size)
            )

        input_feed = {}
        for i in range(encoder_size):
            input_feed[self.encoder_inputs[i].name] = encoder_inputs[i]
        for i in range(decoder_size):
            input_feed[self.decoder_inputs[i].name] = decoder_inputs[i]
            input_feed[self.decoder_weights[i].name] = decoder_weights[i]

        # 理论上decoder inputs和decoder target都是n位
        # 但是实际上decoder inputs分配了n+1位空间
        # 不过inputs是第[0, n)，而target是[1, n+1)，刚好错开一位
        # 最后这一位是没东西的，所以要补齐最后一位，填充0
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        if not forward_only:
            output_feed = [
                self.updates[bucket_id],
                self.gradient_norms[bucket_id],
                self.losses[bucket_id]
            ]
            output_feed.append(self.outputs[bucket_id][i])
        else:
            output_feed = [self.losses[bucket_id]]
            for i in range(decoder_size):
                output_feed.append(self.outputs[bucket_id][i])

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], outputs[3:]
        else:
            return None, outputs[0], outputs[1:]

    def get_batch_data(self, bucket_dbs, bucket_id):
        data = []
        data_in = []
        bucket_db = bucket_dbs[bucket_id]
        for _ in range(self.batch_size):
            ask, answer = bucket_db.random()
            data.append((ask, answer))
            data_in.append((answer, ask))
        return data, data_in

    def get_batch(self, bucket_dbs, bucket_id, data):
        encoder_size, decoder_size = self.buckets[bucket_id]
        # bucket_db = bucket_dbs[bucket_id]
        encoder_inputs, decoder_inputs = [], []
        for encoder_input, decoder_input in data:
            # encoder_input, decoder_input = random.choice(data[bucket_id])
            # encoder_input, decoder_input = bucket_db.random()
            encoder_input = data_utils.sentence_indice(encoder_input)
            decoder_input = data_utils.sentence_indice(decoder_input)
            # Encoder
            encoder_pad = [data_utils.PAD_ID] * (
                encoder_size - len(encoder_input)
            )
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
            # Decoder
            decoder_pad_size = decoder_size - len(decoder_input) - 2
            decoder_inputs.append(
                [data_utils.GO_ID] + decoder_input +
                [data_utils.EOS_ID] +
                [data_utils.PAD_ID] * decoder_pad_size
            )
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
        # batch encoder
        for i in range(encoder_size):
            batch_encoder_inputs.append(np.array(
                [encoder_inputs[j][i] for j in range(self.batch_size)],
                dtype=np.int32
            ))
        # batch decoder
        for i in range(decoder_size):
            batch_decoder_inputs.append(np.array(
                [decoder_inputs[j][i] for j in range(self.batch_size)],
                dtype=np.int32
            ))
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for j in range(self.batch_size):
                if i < decoder_size - 1:
                    target = decoder_inputs[j][i + 1]
                if i == decoder_size - 1 or target == data_utils.PAD_ID:
                    batch_weight[j] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights
