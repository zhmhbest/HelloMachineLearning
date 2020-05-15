from com.zhmh.tf import set_log_level, TensorBoard
import tensorflow as tf
set_log_level(2)


if __name__ == '__main__':
    a = tf.constant(5.0, name="a")
    b = tf.constant(6.0, name="b")
    c = tf.add(a, b, name='c')

    tb = TensorBoard()
    with tf.Session() as sess:
        tb.save(sess.graph)
    tb.board()
