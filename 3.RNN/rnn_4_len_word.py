# Copyright 2017 kairos03. All Right Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ===============================================================
import tensorflow as tf
import numpy as np
import random


def next_batch(batch_size):
    """ make next batch input

    Return:
        bx: batch_x input
        by: batch_y label
    """
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    alpha_list = [alphabet[n] for n in range(len(alphabet))]
    num_dict = {n: i for i, n in enumerate(alpha_list)}
    seq_data = ['word', 'wood', 'love', 'life', 'deep', 'sexy', 'mood',
                'dive', 'cold', 'cool', 'kiss', 'ship', 'room', 'mart']

    bx = []
    by = []

    # randomize
    random.shuffle(seq_data)

    for i in range(batch_size):
        # match char to num
        # 'abcd' = [0 1 2 3]
        num_word = [num_dict[w] for w in seq_data[i]]

        # make input and label
        # '0 1 2 3' => x:[0 1 2], y:[3]
        x = num_word[:-1]
        y = num_word[-1]

        # one_hot to x
        # [0 1 2]:
        # [[1. 0. 0. 0. ... ],
        #  [0. 1. 0. 0. ... ],
        #  [0. 0. 1. 0. ... ]]
        bx.append(np.eye(len(num_dict))[x])
        # sparse_softmax_cross_entropy_with_logits are not need to one_hot
        by.append(y)

    return bx, by


def train():

    learning_rate = 0.01
    n_hidden = 128
    total_epoch = 1000
    n_step = 3
    n_input = n_class = 26
    batch_size = 13
    logdir = './rnn_4_len_word/lr_%s_hd_%s_ep_%s' % (learning_rate, n_hidden, total_epoch)

    # weight
    def weight_var(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # bias
    def bias_var(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def variable_summery(name, var):
        with tf.name_scope(name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.histogram('histogram', var)

    # 3.RNN model
    # placeholder
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, n_step, n_input])
    with tf.name_scope('output'):
        y_ = tf.placeholder(tf.int32, [None])

    # var
    with tf.name_scope('hidden'):
        w = weight_var([n_hidden, n_class])
        b = bias_var([n_class])
        variable_summery('weight', w)
        variable_summery('bias', b)

    # cell
    def rnn_cell(n_hidden, cell_type=tf.nn.rnn_cell.BasicLSTMCell, keep_prob=1, name='rnn_cell'):
        cell = cell_type(n_hidden)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
        return cell

    with tf.name_scope('multi_cell'):
        cell1 = rnn_cell(n_hidden, keep_prob=0.9, name='cell1')
        cell2 = rnn_cell(n_hidden, name='cell2')

        multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

        outputs, states = tf.nn.dynamic_rnn(multi_cell, x, dtype=tf.float32)

        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = outputs[-1]
        model = tf.matmul(outputs, w) + b
        variable_summery('act', model)

    with tf.name_scope('matrices'):
        with tf.name_scope('xent'):
            xent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=y_), name='xent')
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(xent)
        # accuracy
        with tf.name_scope('accuracy'):
            correct = tf.equal(tf.cast(tf.argmax(model, 1), tf.int32), y_)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.summary.scalar('xent', xent)
        tf.summary.scalar('accuracy', accuracy)

    # merge summery
    sess = tf.Session()
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(logdir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(logdir + '/test')
    sess.run(tf.global_variables_initializer())

    def feed_dict(train):
        if train:
            xs, ys = next_batch(10)
        else:
            xs, ys = next_batch(14)
        return {x: xs, y_: ys}

    for i in range(total_epoch):
        if i % 10 == 0:  # recode summaries and test-set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Accuracy at step {}: {}'.format(i, acc))
        else:  # recode train set summaries and training
            if i % 100 == 99:  # recode execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, optimizer],
                                      feed_dict=feed_dict(True),
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run matadata for', i)
            else:  # recode summary
                summary, _ = sess.run([merged, optimizer], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()


if __name__ == '__main__':
    train()
    # x, y = next_batch(1)
    # print(x)
    # print(y)

