import tensorflow as tf
from com.zhmh.tf import generate_random_data, next_batch
from com.zhmh.tf import TensorBoard
from com.zhmh.tf import set_log_level
set_log_level(2)
board = TensorBoard()


"""
    基本设置
"""
train_times = 5000
learning_rate = 0.001
batch_size = 8
dataset_size = 128


"""
    生成模拟数据集
    x1 x2 y
"""
DATA_X, DATA_Y = generate_random_data(dataset_size, 2, 1)


"""
    定义神经网络的参数，输入和输出节点
    x1  a_{11}
        a_{12}  y
    x2  a_{13}
"""
input_x = tf.placeholder(tf.float32, shape=(None, 2), name="x_input")
input_y = tf.placeholder(tf.float32, shape=(None, 1), name='y_input')
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1, name='init_w1'), name='w1')
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1, name='init_w2'), name='w2')
a = tf.matmul(input_x, w1, name='a')
y = tf.matmul(a, w2, name='y')


"""
    交叉熵（Cross-Entropy）
    - tf.clip_by_value: 将一个张量中的数值限制在一个范围之内。
    - tf.log: 对张量中所有元素依次求对数
    - *: 矩阵对应位置直接相乘
    - matmul: 矩阵乘法
"""
cross_entropy = -tf.reduce_mean(
    input_y * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) +
    (1 - input_y) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)),
    name='cross_entropy'
)
loss = cross_entropy
train_adam = tf.train.AdamOptimizer(learning_rate).minimize(loss)


"""
    训练
"""
with tf.Session() as sess:
    # 初始化全部变量OP
    tf.global_variables_initializer().run()

    # 未经训练的参数取值。
    print("Before w1:\n", sess.run(w1))
    print("Before w2:\n", sess.run(w2))
    print()

    # 训练模型
    for i in range(1, 1 + train_times):
        batch_X, batch_Y = next_batch(DATA_X, DATA_Y, i, dataset_size, batch_size)
        sess.run(train_adam, feed_dict={input_x: batch_X, input_y: batch_Y})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={input_x: DATA_X, input_y: DATA_Y})
            print("训练%d次后，交叉熵为%g。" % (i, total_cross_entropy))
    # end for

    # 训练后的参数取值。
    print()
    print("After w1:\n", sess.run(w1))
    print("After w2:\n", sess.run(w2))

    # 保存模型
    board.save(sess.graph)
# end with


# 启动TensorBoard
board.board()
