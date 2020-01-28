import tensorflow as tf


print("""
========================================
    一般情况下，张量创建在默认图上
========================================""".strip())
g0 = tf.get_default_graph()  # 获取默认的图
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result1 = a + b
result2 = tf.add(a, b)
print(a.graph is g0)
print(b.graph is g0)
print(result1.graph is g0)
print(result2.graph is g0)
print("g0 is default?", tf.get_default_graph() is g0)


print("""
========================================
    自定义图 g1
========================================""".strip())
g1 = tf.Graph()
with g1.as_default():
    print("g1 is default?", tf.get_default_graph() is g1)
    # g1中定义变量，并赋值
    tf.get_variable("my_var", initializer=tf.zeros_initializer(), shape=[2, 3])
# end with(graph)


print("退出with后默认图又自动改变为g0")
print("g0 is default?", tf.get_default_graph() is g0)


print("""
========================================
    自定义图 g2
========================================""".strip())
g2 = tf.Graph()
with g2.as_default():
    print("g2 is default?", tf.get_default_graph() is g2)
    # g2中定义变量，并赋值
    tf.get_variable("my_var", initializer=tf.ones_initializer(), shape=[3, 2])
    # 运行会话
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        with tf.variable_scope("", reuse=True):
            print("g2 my_var", sess.run(tf.get_variable("my_var")))
    # end with(sess)
# end with(graph)


print("退出with后默认图又自动改变为g0")
print("g0 is default?", tf.get_default_graph() is g0)


print("""
========================================
    with(graph) 外运行会话，若不使用默认图，需指定graph
========================================""".strip())
with tf.Session(graph=g1) as sess:
    print("g1 is default?", tf.get_default_graph() is g1)
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print("g1 my_var", sess.run(tf.get_variable("my_var")))
# end with(sess)
