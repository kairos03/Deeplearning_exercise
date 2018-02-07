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
    mnist = input_data.read_data_sets('./mnist/data', one_hot=True)

    sess = tf.InteractiveSession()

    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, [None, 784])
        y_ = tf.placeholder(tf.float32, [None, 10])
        keep_prob = tf.placeholder(tf.float32)

    def summary(var):
        tf.summary.scalar('mean', tf.reduce_mean(var))
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

    def gen_weight(shape):
        weight = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(weight)

    def gen_bias(shape):
        bias = tf.constant(0.1, shape=shape)
        return tf.Variable(bias)

    def dense_layer(input_tensor, input_shape, output_shape, name, act=tf.nn.relu, dropout=keep_prob):
        with tf.name_scope(name):
            with tf.name_scope('weight'):
                w = gen_weight([input_shape, output_shape])
                summary(w)
            with tf.name_scope('bias'):
                b = gen_bias([output_shape])
                summary(b)
            with tf.name_scope('wx_plus_b'):
                preact = tf.matmul(input_tensor, w) + b
            activated = act(preact)
            tf.summary.histogram('activation', activated)
            with tf.name_scope('dropout'):
                dropout = tf.nn.dropout(activated, dropout)
            return dropout

    L1 = dense_layer(x, 784, 1024, 'L1')
    L2 = dense_layer(L1, 1024, 512, 'L2')
    y = dense_layer(L2, 512, 10, 'L3', act=tf.identity)

    with tf.name_scope('Matrices'):
        with tf.name_scope('cross_entropy'):
            diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
            with tf.name_scope('total'):
                cross_entropy = tf.reduce_mean(diff)
        tf.summary.scalar('cross_entropy', cross_entropy)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

    # merge summery
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
    tf.global_variables_initializer().run()

    def feed_dict(train):
        if train or FLAGS.fake_data:
            xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
            xs = xs.reshape(-1, 784)
            k = FLAGS.dropout
        else:
            xs, ys = mnist.test.images, mnist.test.labels
            xs = xs.reshape(-1, 784)
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}

    for i in range(FLAGS.max_steps):
        if i % 10 == 0:  # recode summaries and test-set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Accuracy at step {}: {}'.format(i, acc))
        else:  # recode train set summaries and training
            if i % 100 == 99:  # recode execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict=feed_dict(True),
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run matadata for', i)
            else:  # recode summary
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
