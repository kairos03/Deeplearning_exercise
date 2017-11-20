# Copyright 2017 kairos03. All Right Reserved.

import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import os

import image_data

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('../mnist/data', one_hot=True)
gogh = image_data.read_data_sets(data_path='./gogh_test')

# hyperparameter
total_epoch = 5000
batch_size = 2
d_learning_rate = 1e-5
g_learning_rate = 1e-3

# model var
n_hidden = 4096
n_input = 100 * 100 * 1
n_noise = 4096

# placeholder
with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, [None, n_input])
    Z = tf.placeholder(tf.float32, [None, n_noise])


def generator(noise):
    with tf.variable_scope('generator'):
        hidden = tf.layers.dense(noise, n_hidden, activation=tf.nn.relu)
        output = tf.layers.dense(hidden, n_input, activation=tf.nn.sigmoid)
    return output


def discriminator(inputs, reuse=None):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()

        hidden = tf.layers.dense(inputs, n_hidden, activation=tf.nn.relu)
        output = tf.layers.dense(hidden, 1, activation=None)    # TODO why??
    return output


def get_noise(batch_size, n_noise):
    return np.random.uniform(-1., 1., size=[batch_size, n_noise])

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
    train_D = tf.train.AdamOptimizer(d_learning_rate).minimize(D_loss, var_list=vars_D)
    train_G = tf.train.AdamOptimizer(g_learning_rate).minimize(G_loss, var_list=vars_G)

# train
with tf.Session() as sess:
    # merge summery
    merged = tf.summary.merge_all()
    name = 'd_{}_g_{}_t_{}'.format(d_learning_rate, g_learning_rate, time.time())
    train_writer = tf.summary.FileWriter('./train/{}'.format(name), sess.graph)
    tf.global_variables_initializer().run()

    total_batch = int(gogh.num_images/batch_size)
    loss_val_D, loss_val_G = 0, 0

    os.mkdir('samples2/{}'.format(name))

    for epoch in range(total_epoch):
        # b_xs, noise = None, None
        # or i in range(total_batch):
        # b_xs, _ = mnist.train.next_batch(1)
        # b_xs = gogh.next_batch(batch_size)
        b_xs = gogh.next()
        b_xs = np.reshape(b_xs, [-1, n_input])
        noise = get_noise(batch_size, n_noise)

        # train each D and G
        _, loss_val_D = sess.run([train_D, D_loss], feed_dict={X: b_xs, Z: noise})
        _, loss_val_G = sess.run([train_G, G_loss], feed_dict={Z: noise})

        print('Epoch:', '%04d' % epoch,
              'D loss: {:.4}'.format(loss_val_D),
              'G loss: {:.4}'.format(loss_val_G))

        if epoch == 0 or (epoch+1) % 10 == 0:
            # add summary
            summary = sess.run(merged, feed_dict={X: b_xs, Z: noise})
            train_writer.add_summary(summary, epoch)

        if epoch == 0 or (epoch + 1) % 200 == 0:
            # sample_size = 10
            sample_size = 2
            noise = get_noise(sample_size, n_noise)
            samples = sess.run(G, feed_dict={Z: noise})

            fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

            for i in range(sample_size):
                ax[0][i].set_axis_off()
                ax[1][i].set_axis_off()

                ax[0][i].imshow(np.reshape(gogh.images[i], (100, 100)))
                ax[1][i].imshow(np.reshape(samples[i], (100, 100)))
            plt.savefig('samples2/{}/{}.png'.format(name, str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)

    print('최적화 완료!')
