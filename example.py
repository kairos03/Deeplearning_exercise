import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

#####
# ===== Model Define ===== #
#####

# W1 [3 3 1 32] -> [3 3]: kernel, 1: input_x's feature, 32: num of filters
# L1 Conv shape = (?, 28, 28, 32)
#    Pool       ->(?, 14, 14, 32)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
L1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# L1 = tf.nn.dropout(L1, keep_prob)

# W2 [3 3 32 64] -> [3 3]: kernel, 32: input_x's feature 64: num of filters
# L2 Conv shape = (?, 14, 14, 64)
#    Pool       ->(?, 7, 7, 64)

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# L2 = tf.nn.dropout(L2, keep_prob)

# FC 7x7x64 -> 1024
W3 = tf.Variable(tf.random_normal([7 * 7 * 64, 1024], stddev=0.01))
L3 = tf.reshape(L2, [-1, 7 * 7 * 64])
L3 = tf.matmul(L3, W3)
L3 = tf.nn.relu(L3)
L3 = tf.nn.dropout(L3, keep_prob)

# output 1024 -> 10
W4 = tf.Variable(tf.random_normal([1024, 10], stddev=0.01))
model = tf.matmul(L3, W4)

# cost and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdadeltaOptimizer(0.01).minimize(cost)

#####
# ===== Model Learning ===== #
#####
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(20000):
    total_cost = 0
    prev_avg_cost = 100

    for i in range(total_batch):

        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # image reshape [28, 28, 1]
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)

        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={X: batch_xs,
                                          Y: batch_ys,
                                          keep_prob: 0.7})

        total_cost += cost_val

    cur_avg_cost = total_cost / total_batch
    if epoch%10 == 0:
        print('Epoch: ', '%05d' % epoch,
              'Avg. cost =', '{:.6f}'.format(cur_avg_cost))

    # cost 변화가 1e-4보다 작으면 학습 종료
    if abs(prev_avg_cost - cur_avg_cost) < 1e-4:
        break

    prev_avg_cost = cur_avg_cost

print('complete!')

# result
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print("accuracy: ", sess.run(accuracy,
                             feed_dict={X: mnist.test.images.reshape(-1,28,28,1),
                                        Y: mnist.test.labels,
                                        keep_prob: 1}))
# model save
tf.train.Saver().save(sess, 'cnn_model')
