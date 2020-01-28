import tensorflow as tf


print("""
========================================
    定义变量 Variable
========================================""".strip())
my_var1 = tf.Variable(99, name='my_var1')
my_var2 = tf.Variable(tf.random_normal([2, 3], mean=10, stddev=2), name='my_var2')
print(my_var1)
print(my_var2)


print("""
========================================
    定义变量 get_variable
========================================""".strip())
print("""
当设置reuse=True时，get_variable可以防止重复定义
该方法定义变量只能使用initializer方法初始化
    tf.constant_initializer()
    tf.random_normal_initializer()
    tf.truncated_normal_initializer()
    tf.random_uniform_initializer()
    tf.uniform_unit_scaling_initializer()
    tf.zeros_initializer()
    tf.ones_initializer()
----------------------------------------
""".strip())
my_var3 = tf.get_variable('my_var3', initializer=tf.zeros_initializer(), shape=[2])
my_var4 = tf.get_variable('my_var4', initializer=tf.ones_initializer(), shape=[3])
print(my_var3)
print(my_var4)


print("""
========================================
    使用变量
========================================""".strip())
with tf.Session() as sess:
    # 使用前，必须初始化变量
    sess.run(my_var1.initializer)
    sess.run(my_var3.initializer)

    # 获得变量的值
    print(sess.run(my_var1))
    # print(sess.run(my_var2))  # my_var2未被初始化，使用会报错。

    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("my_var3")))
        # print(sess.run(tf.get_variable("my_var4")))  # my_var4未被初始化，使用会报错。
# end with(sess)


print("""
========================================
    使用变量 global_variables_initializer
========================================""".strip())
with tf.Session() as sess:
    # 初始化全部变量
    tf.global_variables_initializer().run()
    # sess.run(tf.global_variables_initializer())

    # 获得变量的值
    print(sess.run(my_var1))
    print(sess.run(my_var2))
    print()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("my_var3")))
        print(sess.run(tf.get_variable("my_var4")))
# end with(sess)


print("""
========================================
    变量空间
========================================""".strip())
with tf.variable_scope("space1"):
    # 创建变量 space1.my_var1
    tf.get_variable('my_var1', initializer=tf.zeros_initializer(), shape=[1])

with tf.variable_scope("space2"):
    # 创建变量 space2.my_var1
    tf.get_variable('my_var1', initializer=tf.ones_initializer(), shape=[1])

with tf.Session() as sess:
    # 初始化全部变量
    tf.global_variables_initializer().run()

    with tf.variable_scope("space1", reuse=True):
        print(sess.run(tf.get_variable("my_var1")))

    with tf.variable_scope("space2", reuse=True):
        print(sess.run(tf.get_variable("my_var1")))
# end with(sess)
