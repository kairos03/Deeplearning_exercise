import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./mnist/data/', one_hot=True)

#
learning_rate = 0.001
total_epoch = 200
batch_size = 200

n_input = 28
n_step = 28
n_hidden = 128
n_class = 10


def var_summary(var, name):
    tf.summary.scalar('mean', tf.reduce_mean(var))
    tf.summary.histogram(name, var)


with tf.variable_scope('input'):
    x = tf.placeholder(tf.float32, [None, n_step, n_input])
    y_ = tf.placeholder(tf.float32, [None, n_class])

with tf.variable_scope('cell'):
    W = tf.Variable(tf.truncated_normal([n_hidden, n_class]))
    b = tf.Variable(tf.truncated_normal([n_class]))

    var_summary(W, "weight")
    var_summary(b, "bias")

    cell = tf.nn.rnn_cell.GRUCell(n_hidden)

    outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    outputs = tf.transpose(outputs, [1, 0, 2])
    outputs = outputs[-1]
    model = tf.matmul(outputs, W) + b

    tf.summary.histogram('model', model)

with tf.variable_scope('Matrix'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y_))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    var_summary(cost, "Xent")

sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples/batch_size)

for epoch in range(total_epoch):
    total_cost = 0

    for i in range(total_batch):
        xs, ys = mnist.train.next_batch(batch_size)
        xs = xs.reshape((batch_size, n_step, n_input))

        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={
                                   x: xs,
                                   y_: ys
                               })
        total_cost += cost_val

    print("epoch: {:04d}, cost: {:.8f}".format(epoch+1, total_cost/total_batch))

print('FINISH')

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

test_batch_size = len(mnist.test.images)
test_xs = mnist.test.images.reshape(test_batch_size, n_step, n_input)
test_ys = mnist.test.labels

print('정확도:', sess.run(accuracy,
                       feed_dict={x: test_xs, y_: test_ys}))
