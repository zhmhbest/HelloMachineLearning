from com.zhmh.tf import set_log_level
import tensorflow as tf
set_log_level(2)


# Tensor
a = tf.constant(1)
b = tf.constant(2)
result = tf.add(a, b)
print(result)


# Session
with tf.Session() as sess:
    print(sess.run(result))
