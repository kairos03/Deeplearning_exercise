# Copyright 2017 kairos03. All Right Reserved.

import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from random import shuffle

import image_data

# input
gogh = image_data.read_data_sets(data_path='../0.data_set/gogh_test2')

# hyperparameter
total_epoch = 60000
batch_size = 10
learning_rate = 1e-4
keep_prob = 0.5

# model var
i_128 = [128, 128, 3]
g_128 = [2048, 1024, 512, 256, 128, 64]
d_128 = [64, 128, 256, 512, 1024]

# n_hidden = 4096
n_input = i_128
g_filters = g_128
d_filters = d_128
n_noise = 1 * 1 * 100


# placeholder
with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, [None, n_input[0], n_input[1], n_input[2]])
    Z = tf.placeholder(tf.float32, [None, n_noise])

    tf.summary.image('inputs', X, 12)


def lrelu(x, alpha):
    with tf.name_scope('lrelu'):
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


def make_conv2d_transpose(inputs, filters, ksize=7, strides=2, padding='SAME', activation=lrelu, alpha=0.2):
    layer = tf.layers.conv2d_transpose(inputs, filters=filters, kernel_size=ksize, strides=strides, padding=padding)
    layer = activation(tf.layers.batch_normalization(layer), alpha)
    return layer


def make_conv2d(inputs, filters, ksize=5, strides=2, padding='SAME', activation=lrelu, alpha=0.2):
    layer = tf.layers.conv2d(inputs, filters=filters, kernel_size=ksize, strides=strides, padding=padding)
    if activation is not None:
        layer = activation(tf.layers.batch_normalization(layer), alpha)
    return layer


def generator(noise):
    with tf.variable_scope('generator'):
        # input [None, 100]
        # 4 x 4 x n
        inputs = tf.layers.dense(noise, g_filters[0] * 4 * 4)
        reshape = tf.reshape(inputs, [-1, 4, 4, g_filters[0]])
        conv = tf.nn.relu(tf.layers.batch_normalization(reshape))

        layer_cnt = 0
        for f in g_filters[1:]:
            conv = make_conv2d_transpose(conv, f)

        conv = make_conv2d_transpose(conv, n_input[2], strides=1)
        conv = tf.layers.dropout(conv, keep_prob)
        tf.summary.image('output', conv, 8)

    return conv


def discriminator(inputs, reuse=None):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()

        # input
        conv = inputs

        for f in d_filters:
            conv = make_conv2d(conv, filters=f)

        # reshape
        reshaped = tf.reshape(conv, [-1, 4 * 4 * d_filters[-1]])
        output = tf.layers.dense(reshaped, 1, activation=None)
    return output


def get_noise(batch_size, n_noise):
    return np.random.normal(-1., 1., size=[batch_size, n_noise])


# model
G = generator(Z)
D_real = discriminator(X)
D_gene = discriminator(G, reuse=True)

# loss
with tf.name_scope('loss'):
    D_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real,
                                                labels=tf.ones_like(D_real)))
    D_loss_gene = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_gene,
                                                labels=tf.zeros_like(D_gene)))
    D_loss = D_loss_real + D_loss_gene

    G_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_gene,
                                                labels=tf.ones_like(D_gene)))

    tf.summary.scalar('D_loss', D_loss)
    tf.summary.scalar('G_loss', G_loss)


def var_summary(var):
    tf.summary.histogram('histogram', var)
    mean = tf.reduce_mean(var)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)


# vars
vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
with tf.name_scope('D_vars'):
    for v in vars_D:
        var_summary(v)
with tf.name_scope('G_vars'):
    for v in vars_G:
        var_summary(v)

# optmizer
with tf.name_scope('train'):
    train_D = tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list=vars_D)
    train_G = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list=vars_G)

# train
with tf.Session() as sess:
    # merge summery
    merged = tf.summary.merge_all()
    name = 'lr_{}_t_{}'.format(learning_rate, time.time())
    train_writer = tf.summary.FileWriter('./train/{}'.format(name), sess.graph)
    tf.global_variables_initializer().run()

    total_batch = int(gogh.num_images/batch_size)
    loss_val_D, loss_val_G = 0, 0

    for epoch in range(total_epoch):

        b_xs = gogh.next()
        shuffle(b_xs)
        b_xs = np.reshape(b_xs, [-1, n_input[0], n_input[1], n_input[2]])
        noise = get_noise(batch_size, n_noise)

        # train each D and G
        _, loss_val_D = sess.run([train_D, D_loss], feed_dict={X: b_xs, Z: noise})
        _, loss_val_G = sess.run([train_G, G_loss], feed_dict={Z: noise})

        if epoch == 0 or (epoch+1) % 100 == 0:
            print('Epoch:', '%04d' % epoch,
                  'D loss: {:.05}'.format(loss_val_D),
                  'G loss: {:.05}'.format(loss_val_G))
            # add summary
            summary = sess.run(merged, feed_dict={X: b_xs, Z: noise})
            train_writer.add_summary(summary, epoch)

    print('최적화 완료!')
