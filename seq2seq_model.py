"""Sequence-to-sequence model with an attention mechanism."""
# see https://www.tensorflow.org/versions/r0.10/tutorials/seq2seq/index.html
# compare https://github.com/tflearn/tflearn/blob/master/examples/nlp/seq2seq_example.py
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell
from tensorflow.contrib.legacy_seq2seq import embedding_attention_seq2seq, model_with_buckets

# vocab_size = 256  # We are lazy, so we avoid fency mapping and just use one *class* per character/byte
# target_vocab_size = vocab_size
# learning_rate = 0.1
# buckets = [(50, 50)]  # our input and response words can be up to 10 characters long
# PAD = [0]  # fill words shorter than 10 characters with 'padding' zeroes
# batch_size = 10  # for parallel training (later)

# input_data = [map(ord, "hello") + PAD * 5] * batch_size
# target_data = [map(ord, "world") + PAD * 5] * batch_size
# target_weights = [[1.0] * 6 + [0.0] * 4] * batch_size  # mask padding. todo: redundant --


# EOS='\n' # end of sequence symbol todo use how?
# GO=1		 # start symbol 0x01 todo use how?


class BabySeq2Seq(object):
    def __init__(self, source_vocab_size, target_vocab_size, buckets, size, num_layers, batch_size):
        self.buckets = buckets
        self.batch_size = batch_size
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size

        cell = single_cell = GRUCell(size)
        if num_layers > 1:
            cell = MultiRNNCell([single_cell] * num_layers)

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return embedding_attention_seq2seq(
                encoder_inputs, decoder_inputs, cell,
                num_encoder_symbols=source_vocab_size,
                num_decoder_symbols=target_vocab_size,
                embedding_size=size,
                feed_previous=do_decode)

        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in range(buckets[-1][0]):  # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
        for i in range(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))

        # Our targets are decoder inputs shifted by one. OK
        targets = [self.decoder_inputs[i + 1] for i in range(len(self.decoder_inputs) - 1)]
        self.outputs, self.losses = model_with_buckets(
            self.encoder_inputs, self.decoder_inputs, targets,
            self.target_weights, buckets,
            lambda x, y: seq2seq_f(x, y, False))
        tf.summary.scalar("losses", self.losses[0])
        self.summary_ops = tf.summary.merge_all()
        # Gradients update operation for training the model.
        params = tf.trainable_variables()
        self.updates = []
        for b in range(len(buckets)):
            self.updates.append(tf.train.AdamOptimizer().minimize(self.losses[b]))

        self.saver = tf.train.Saver(tf.global_variables())

    def step(self, session, encoder_inputs, decoder_inputs, target_weights, test):
        bucket_id = 0  # todo: auto-select
        encoder_size, decoder_size = self.buckets[bucket_id]

        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                             " %d != %d." % (len(target_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        if not test:
            output_feed = [self.updates[bucket_id], self.losses[bucket_id], self.summary_ops]
        else:
            output_feed = [self.losses[bucket_id]]  # Loss for this batch.
            for l in range(decoder_size):  # Output logits.
                output_feed.append(self.outputs[bucket_id][l])

        outputs = session.run(output_feed, input_feed)
        if not test:
            return outputs[0], outputs[1], outputs[2]  # Gradient norm, loss, summary_ops
        else:
            return outputs[0], outputs[1:]  # loss, outputs.


# def decode(bytes):
#     return "".join(map(chr, bytes)).replace('\x00', '').replace('\n', '')


def test(model, session, input_data, target_data, target_weights):
    eval_loss, outputs = model.step(session, input_data, target_data, target_weights, test=True)
    outputs = np.array(outputs).transpose([1, 0, 2])
    return eval_loss, np.argmax(outputs, axis=2)[:, :-1]
    # words = np.argmax(outputs, axis=2)  # shape (10, 10, 256)
    # word = decode(words[0])
    # print("step %d, perplexity %f, output: hello %s?" % (step, perplexity, word))
    # if word == "world":
    #     print(">>>>> success! hello " + word + "! <<<<<<<")
    #     exit()


# step = 0
# test_step = 1
# with tf.Session() as session:
#     model = BabySeq2Seq(vocab_size, target_vocab_size, buckets, size=10, num_layers=1, batch_size=batch_size)
#     session.run(tf.initialize_all_variables())
#     while True:
#         model.step(session, input_data, target_data, target_weights, test=False)  # no outputs in training
#         if step % test_step == 0:
#             test()
#         step = step + 1
