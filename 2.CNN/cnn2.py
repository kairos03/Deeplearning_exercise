# 절대 임포트 설정
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 필요한 라이브러리들을 임포트
import argparse
import sys
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


def train():
    # import data
    mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

    sess = tf.InteractiveSession()

    # input
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

    # input reshape
    with tf.name_scope('input_reshape'):
        tf.summary.image('input', x, 10)

    # weight
    def weight_var(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # bias
    def bias_var(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def variable_summery(var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def conv_layer(input_tensor, filter_dim, layer_name,
                   conv_strides=[1, 1, 1, 1], conv_padding="SAME",
                   max_pool_ksize=[1, 2, 2, 1], max_pool_strides=[1, 2, 2, 1], max_pool_padding="SAME",
                   act=tf.nn.relu):
        with tf.name_scope(layer_name):
            with tf.name_scope('filter'):
                filter_weight = weight_var(filter_dim)
                variable_summery(filter_weight)
            with tf.name_scope('conv'):
                conv = tf.nn.conv2d(input_tensor, filter_weight, strides=conv_strides, padding=conv_padding)
                tf.summary.histogram('conv', conv)
            activations = act(conv, name='activation')
            tf.summary.histogram('activations', activations)
            with tf.name_scope('max_pool'):
                max_pool = tf.nn.max_pool(activations, max_pool_ksize, max_pool_strides, max_pool_padding)
                tf.summary.histogram('max_pool', max_pool)
            return max_pool

    def dense_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                weights = weight_var([input_dim, output_dim])
                variable_summery(weights)
            with tf.name_scope('bias'):
                biases = bias_var([output_dim])
                variable_summery(biases)
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
            return activations

    conv1 = conv_layer(x, [3, 3, 1, 32], 'conv1', conv_padding='SAME')
    conv2 = conv_layer(conv1, [3, 3, 32, 64], 'conv2', conv_padding='SAME')

    with tf.name_scope('reshape'):
        resh = tf.reshape(conv2, [-1, 7 * 7 * 64], name='reshape')
        tf.summary.histogram('reshape', resh)

    fc3 = dense_layer(resh, 7 * 7 * 64, 1024, 'fc3')

    # dropout
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(fc3, keep_prob)

    y = dense_layer(dropped, 1024, 10, 'fc4', tf.identity)

    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # merge summery
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir+'/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir+'/test')
    tf.global_variables_initializer().run()

    def feed_dict(train):
        if train or FLAGS.fake_data:
            xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
            xs = xs.reshape(-1, 28, 28, 1)
            k = FLAGS.dropout
        else:
            xs, ys = mnist.test.images, mnist.test.labels
            xs = xs.reshape(-1, 28, 28, 1)
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}

    for i in range(FLAGS.max_steps):
        if i % 10 == 0:  # recode summaries and test-set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Accuracy at step {}: {}'.format(i, acc))
        else:   # recode train set summaries and training
            if i % 100 == 99:   # recode execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict=feed_dict(True),
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run matadata for', i)
            else:   # recode summary
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data', nargs='?', const=True, type=bool, default=False,
                        help='If true, uses fake_data for unit testing.')
    parser.add_argument('--max_steps', type=int, default=10000,
                        help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--dropout', type=float, default=0.9,
                        help='Keep probability for training dropout.')
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/mnist/logs/mnist_with_summaries',
                        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)