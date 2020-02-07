import tensorflow as tf
import numpy as np

M = np.array([
    [[1], [-1], [0]],
    [[-1], [2], [1]],
    [[0], [2], [-2]]
], dtype=np.float).reshape((1, 3, 3, 1))

print(M, M.shape)

input_x = tf.placeholder(tf.float32, [1, None, None, 1])

filter_weight = tf.get_variable('weights', [2, 2, 1, 1], initializer=tf.constant_initializer([
    [1, -1],
    [0, 2]
]))
filter_biases = tf.get_variable('biases', [1], initializer=tf.constant_initializer(1))

conv = tf.nn.conv2d(input_x, filter_weight, strides=[1, 2, 2, 1], padding='SAME')
bias = tf.nn.bias_add(conv, filter_biases)

pool = tf.nn.avg_pool(input_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    convoluted_M = sess.run(bias, feed_dict={input_x: M})
    pooled_M = sess.run(pool, feed_dict={input_x: M})

    print("convoluted_M: \n", convoluted_M)
    print("pooled_M: \n", pooled_M)
