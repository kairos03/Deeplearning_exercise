import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

# placeholder
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# model
# L1 (Conv)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32]))
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding="SAME")
L1 = tf.nn.relu(L1)
#L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# L2 (Conv)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64]))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding="SAME")
L2 = tf.nn.relu(L2)
#L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# L3 (FC)
W3 = tf.Variable(tf.random_normal([28 * 28 * 64, 1024]))
L3 = tf.reshape(L2, [-1, 28 * 28 * 64])
L3 = tf.matmul(L3, W3)
L3 = tf.nn.relu(L3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

# L4 (FC)
W4 = tf.Variable(tf.random_normal([1024, 10]))
model = tf.matmul(L3, W4)

# cost and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))
optimizer = tf.train.AdadeltaOptimizer(0.01).minimize(cost)

# train
# init
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# batch
batch_size = 100
total_batch_size = int(mnist.train.num_examples / batch_size)

# training
print("Start!")
for epoch in range(500):
    total_cost = 0
    for i in range(total_batch_size):

        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)

        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={X: batch_xs,
                                          Y: batch_ys,
                                          keep_prob: 0.7})
        total_cost += cost_val

    if epoch % 10 == 0:
        print('epoch: {:05d}, avg_cost: {:.6f}'.format(epoch, (total_cost/total_batch_size)))

    # model save
    tf.train.Saver().save(sess, 'cnn_model')

print("Complete!")

# validate
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print("accuracy: ", sess.run([accuracy],
                             feed_dict={X: mnist.test.images.reshape(-1, 28, 28, 1),
                                        Y: mnist.test.labels,
                                        keep_prob: 1}))
