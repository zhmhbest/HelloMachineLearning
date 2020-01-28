import tensorflow as tf


# Tensor
a = tf.constant(1)
b = tf.constant(2)
result = tf.add(a, b)
print(result)


# Session
with tf.Session() as sess:
    print(sess.run(result))
