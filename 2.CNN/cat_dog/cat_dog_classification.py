# Copyright 2017 kairos03. All Right Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ===============================================================

import tensorflow as tf
from input_data import *


def train():
    # hyper parameter
    learning_rate = 1e-3
    epoch = 1001
    batch_size = 100
    logdir = './log/lr_%s_ep_%s/' % (learning_rate, epoch)

    def variable_summary(var):
        """add summary

        Args:
            var: variable, tf.Variable

        Return:
            nothing
        """
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('stddev', tf.sqrt(tf.reduce_mean(tf.square(var-mean))))
        tf.summary.histogram('histogram', var)

    def variable_weight(shape, stddev=0.01):
        """weight generation with truncated_normal

        Args:
            shape: shape of generated weight variable
            stddev: standard deviation
        Return:
            weight: tf.Variable
        """
        initial = tf.truncated_normal(shape, stddev=stddev)
        return tf.Variable(initial)

    def variable_bias(shape):
        """bias generation with const 0.1

        Args:
            shape: shape of generated bias variable
        Return:
            bias: tf.Variable
        """
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d_layer(input, kernel_dim, name='conv2d',
                     activation=tf.nn.relu,
                     strides=[1, 1, 1, 1], padding='SAME'):
        """conv2d layer

        Args:
            input: input tensor, Tensor
            kernel_dim: kernel for conv2d, Tensor
            name: name of layer, String
            activation: activation function, a kind of activation
            strides: stride for conv2d, 4 dim array
            padding: padding solution for conv2d, 'SAME' or 'VALID'

        Return:
            act: Tensor
        """
        with tf.name_scope(name):
            with tf.name_scope('kernel'):
                kernel = variable_weight(kernel_dim)
                variable_summary(kernel)
            with tf.name_scope('conv'):
                layer = tf.nn.conv2d(input, kernel, strides=strides, padding=padding, name='conv')
                tf.summary.histogram('conv', layer)
            act = activation(layer)
            tf.summary.histogram('activation', act)
            return act

    def max_pool_layer(input, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME', name='max_pool'):
        """max pooling layer

        Args:
            input: Tensor
            ksize: kernel size, 4dim list
            strides: strides, 4dim list
            padding: padding soultion, 'SAME' or 'VALID'
            name: String

        Return:
            max_pool: max pooled Tensor
        """
        with tf.name_scope(name):
            max_pool = tf.nn.max_pool(input, ksize=ksize, strides=strides, padding=padding)
            tf.summary.histogram('max_pool', max_pool)
            return max_pool

    def dense_layer(input, input_dim, output_dim, name='dense', activation=tf.nn.relu):
        """Basic DNN layer

        Args:
            input: Tensor
            input_dim: int
            output_dim: int
            name: String
            activation: activation function

        Return:
            output: Tensor
        """
        with tf.name_scope(name):
            with tf.name_scope('weight'):
                w = variable_weight([input_dim, output_dim])
                variable_summary(w)
            with tf.name_scope('bias'):
                b = variable_bias([output_dim])
                variable_summary(b)
            with tf.name_scope('wx_plus_b'):
                result = tf.matmul(input, w) + b
                tf.summary.histogram('preactivation', result)
            act = activation(result)
            tf.summary.histogram('activation', act)
            return act

    def dropout(input, keep_proba):
        with tf.name_scope('dropout'):
            drop = tf.nn.dropout(input, keep_prob=keep_proba)
            return drop

    ###########
    # model
    ###########

    # input
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 200, 200, 3])
        y_ = tf.placeholder(tf.int32, [None])
        keep_proba = tf.placeholder(tf.float32)

    # layers
    with tf.name_scope('conv1'):
        l1 = conv2d_layer(x, [200, 200, 3, 32], strides=[1, 1, 1, 1], padding='SAME')
        l1 = max_pool_layer(l1, ksize=[1, 5, 5, 1], strides=[1, 5, 5, 1], padding='SAME')

    with tf.name_scope('conv2'):
        l2 = conv2d_layer(l1, [40, 40, 32, 64], strides=[1, 1, 1, 1], padding='SAME')
        l2 = max_pool_layer(l2, ksize=[1, 5, 5, 1], strides=[1, 5, 5, 1], padding='SAME')

    with tf.name_scope('conv3'):
        l3 = conv2d_layer(l2, [8, 8, 64, 128], strides=[1, 1, 1, 1], padding='SAME')
        l3 = max_pool_layer(l3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    l3 = tf.reshape(l3, [-1, 4 * 4 * 128])

    with tf.name_scope('dense1'):
        l4 = dense_layer(l3, 4 * 4 * 128, 1024)
        l4 = dropout(l4, keep_proba=keep_proba)

    with tf.name_scope('dense2'):
        l5 = dense_layer(l4, 1024, 256)
        l5 = dropout(l5, keep_proba=keep_proba)

    with tf.name_scope('ouput'):
        model = dense_layer(l5, 256, 2, activation=tf.identity)

    with tf.name_scope('matrices'):
        with tf.name_scope('xent'):
            xent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=y_))
        tf.summary.scalar('xent', xent)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(xent)
        with tf.name_scope('accuracy'):
            currect = tf.equal(tf.cast(tf.argmax(model, 1), tf.int32), y_)
            accuracy = tf.reduce_mean(tf.cast(currect, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    # init
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    # summaries
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(logdir+'/train', sess.graph)
    test_writer = tf.summary.FileWriter(logdir+'/test')

    def feed_dict(train):
        if train:
            xs, ys = next_batch(batch_size)
            keep_proba = 0.9
        else:
            xs, ys = next_batch(500)
            keep_proba = 1
        return {x: xs, y_: ys, keep_proba: keep_proba}

    # train
    for i in range(epoch):
        if i % 10 == 0:
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Accuracy at step {}: {}'.format(i, acc))
        else:
            if i % 99 == 0:
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
