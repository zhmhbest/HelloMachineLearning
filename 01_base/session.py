import tensorflow as tf

a = tf.constant(1, name="a")
b = tf.constant(2, name="b")
result = tf.add(a, b)


print("""
========================================
    Session
========================================""".strip())
with tf.Session() as sess:
    print(sess.run(result))
    print(result.eval(session=sess))


print("""
========================================
    Session as_default
========================================""".strip())
with tf.Session().as_default():
    print(result.eval())


print("""
========================================
    InteractiveSession = Session as_default
========================================""".strip())
sess = tf.InteractiveSession()
print(result.eval())
sess.close()


print("""
========================================
    Session config
        log_device_placement: 打印设备信息
        allow_soft_placement: GPU异常时，可以调整到CPU执行
========================================""".strip())
with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    print(sess.run(result))


print("""
========================================
    定义形参，在执行的时候再赋具体的值
========================================""".strip())
X = tf.placeholder(tf.float32, shape=(None, 2))  # 样本组数不固定，每组两个
Y = X * 3
with tf.Session() as sess:
    # print(sess.run(Y))  # InvalidArgumentError: 此处x还没有赋值
    print(sess.run(Y, feed_dict={X: [
        [1, 2],
        [2, 3]
    ]}))

    print(sess.run(Y, feed_dict={X: [
        [1, 2],
        [2, 3],
        [5, 3]
    ]}))
