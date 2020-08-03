import numpy as np
import tensorflow as tf
from zhmh.dataset import BatchGenerator, generate_random_data

"""
    生成模拟数据集
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    x_1
        ↘
           y → output
        ↗
    x_2
"""
DATA_SIZE = 128
INPUT_SIZE = 2
OUTPUT_SIZE = 1
# data_all = np.random.rand(DATA_SIZE * (INPUT_SIZE + OUTPUT_SIZE)).reshape(DATA_SIZE, -1)
# x_train = data_all[:, 0:INPUT_SIZE]
# y_train = data_all[:, INPUT_SIZE:(INPUT_SIZE + OUTPUT_SIZE)]
x_train, y_train = generate_random_data(DATA_SIZE, INPUT_SIZE, OUTPUT_SIZE)
print(x_train.shape, y_train.shape)

"""
    定义输入、输出节点
    定义网络
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    x_1        a_{11}
        →(w1)→ a_{12} →(w2)→ y
    x_2        a_{13}
"""
HIDDEN_SIZE = 3

place_x = tf.placeholder(tf.float32, shape=(None, INPUT_SIZE))
place_y = tf.placeholder(tf.float32, shape=(None, OUTPUT_SIZE))

w1 = tf.Variable(tf.random_normal([INPUT_SIZE, HIDDEN_SIZE], stddev=1))
w2 = tf.Variable(tf.random_normal([HIDDEN_SIZE, OUTPUT_SIZE], stddev=1))

a = tf.matmul(place_x, w1)
y = tf.matmul(a, w2)

"""
    定义损失函数
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
"""
loss = tf.losses.mean_squared_error(place_y, y)

"""
    定义优化器
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
"""
LEARNING_RATE = 0.001
train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

"""
    训练
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
"""
BATCH_SIZE = 8
batch = BatchGenerator(x_train, y_train, BATCH_SIZE)
TRAIN_TIMES = batch.count() * 10
print('TRAIN_TIMES:', TRAIN_TIMES)

with tf.Session() as sess:
    # 初始化全部变量OP
    tf.global_variables_initializer().run()

    # 未经训练的参数取值。
    print("Before w1:\n", sess.run(w1))
    print("Before w2:\n", sess.run(w2))
    print()

    # 训练模型
    for i in range(1, 1 + TRAIN_TIMES):
        batch_x, batch_y = batch.next()
        sess.run(train_op, feed_dict={
            place_x: batch_x,
            place_y: batch_y
        })
        if i % 200 == 0:
            loss_value = sess.run(loss, feed_dict={place_x: x_train, place_y: y_train})
            print("训练%d次后，损失为%g。" % (i, loss_value))

    # 训练后的参数取值。
    print()
    print("After w1:\n", sess.run(w1))
    print("After w2:\n", sess.run(w2))
