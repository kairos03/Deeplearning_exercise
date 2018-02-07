import tensorflow as tf

mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir='./data', one_hot=True)


# Add convolution layer
def conv_layer(input, size_in, size_out, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + b)
        return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# Add fully connected layer
def fc_layer(input, size_in, size_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        act = tf.nn.relu(tf.matmul(input, w) + b)
        return act


def main():
    tf.reset_default_graph()
    sess = tf.Session()

    # Setup placeholders, and reshape the data
    x = tf.placeholder(tf.float32, shape=[None, 784])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    y = tf.placeholder(tf.float32, shape=[None, 10])

    # model
    conv1 = conv_layer(x_image, 1, 32)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv2 = conv_layer(pool1, 32, 64)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    flattened = tf.reshape(pool2, [-1, 7 * 7 * 64])

    fc1 = fc_layer(flattened, 7 * 7 * 64, 1024)
    logits = fc_layer(fc1, 1024, 10)

    # loss
    xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y), name="xent")

    # optimizer
    train_step = tf.train.AdamOptimizer(1e-4).minimize(xent)

    # accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # init
    sess.run(tf.global_variables_initializer())

    # add graph
    writer = tf.summary.FileWriter('')
    writer.add_graph(sess.graph)

    # train
    for i in range(2001):
        batch = mnist.train.next_batch(100)

        if i % 500 == 0:
            [train_accuracy] = sess.run([accuracy], feed_dict={x: batch[0], y: batch[1]})
            print("step %d, training accuracy %g" % (i, train_accuracy))

        sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})


if __name__ == '__main__':
    main()
