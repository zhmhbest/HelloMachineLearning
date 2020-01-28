import tensorflow as tf
from com.zhmh.tf.board import TensorBoard

if __name__ == '__main__':
    a = tf.constant(5.0, name="a")
    b = tf.constant(6.0, name="b")
    c = tf.add(a, b, name='c')

    tb = TensorBoard()
    with tf.Session() as sess:
        tb.save(sess.graph)
    tb.board()
