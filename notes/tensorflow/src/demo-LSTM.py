import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from zhmh.dataset import BatchGenerator
from zhmh.dataset.stock_sh1 import load_stock_sh000001_data

"""
    导入数据
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
"""
data_x, data_y = load_stock_sh000001_data()
INPUT_SIZE = data_x.shape[1]
OUTPUT_SIZE = data_y.shape[1]
print(INPUT_SIZE, OUTPUT_SIZE)

"""
    数据预处理
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
"""
# 标准化
std_x, std_y = StandardScaler(), StandardScaler()
data_x = std_x.fit_transform(data_x)
data_y = std_y.fit_transform(data_y)

# 划分数据集
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.3)
TRAIN_SIZE = train_x.shape[0]
TEST_SIZE = test_x.shape[0]
print(TRAIN_SIZE, TEST_SIZE)

"""
    定义网络
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
"""
TIME_STEP = 15
RNN_UNIT_SIZE = 10
BATCH_SIZE = 80
SINGLE_GROUP_SIZE = BATCH_SIZE * TIME_STEP

place_x = tf.placeholder(tf.float32, shape=[None, TIME_STEP, INPUT_SIZE])
place_y = tf.placeholder(tf.float32, shape=[None, TIME_STEP, OUTPUT_SIZE])
weights = [
    tf.Variable(tf.random_normal([INPUT_SIZE, RNN_UNIT_SIZE])),
    tf.Variable(tf.random_normal([RNN_UNIT_SIZE, OUTPUT_SIZE]))
]
biases = [
    tf.Variable(tf.constant(0.1, shape=[RNN_UNIT_SIZE])),
    tf.Variable(tf.constant(0.1, shape=[OUTPUT_SIZE]))
]
after_place = tf.reshape(place_x, [-1, INPUT_SIZE])
input_rnn = tf.reshape(
    tf.matmul(after_place, weights[0]) + biases[0],
    [-1, TIME_STEP, RNN_UNIT_SIZE]  # 将tensor转成3维，作为cell的输入
)
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(RNN_UNIT_SIZE)
init_state = lstm_cell.zero_state(BATCH_SIZE, dtype=tf.float32)
# output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
output_rnn, final_states = tf.nn.dynamic_rnn(
    lstm_cell,
    input_rnn,
    initial_state=init_state,
    dtype=tf.float32
)
y = tf.matmul(
    tf.reshape(output_rnn, [-1, RNN_UNIT_SIZE]),
    weights[1]
) + biases[1]

"""
    损失函数
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
"""
loss = tf.losses.mean_squared_error(tf.reshape(place_y, [-1]), tf.reshape(y, [-1]))
# loss = tf.reduce_mean(
#     tf.square(
#         tf.reshape(y, [-1]) -
#         tf.reshape(place_y, [-1])
#     )
# )

"""
    训练网络
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
"""
train_batch = BatchGenerator(train_x, train_y, SINGLE_GROUP_SIZE)
test_batch = BatchGenerator(test_x, test_y, SINGLE_GROUP_SIZE, TIME_STEP)
TRAIN_TIMES = train_batch.count() * 1
# TRAIN_TIMES = 50
LEARNING_RATE = 0.0006
train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(TRAIN_TIMES):
        # 每次进行训练的时候，每个batch训练batch_size个样本
        batch_x, batch_y = train_batch.next()
        batch_x = batch_x.reshape([-1, TIME_STEP, INPUT_SIZE])
        batch_y = batch_y.reshape([-1, TIME_STEP, OUTPUT_SIZE])
        _, loss_val = sess.run(
            [train_op, loss],
            feed_dict={
                place_x: batch_x,
                place_y: batch_y
            }
        )
        if 0 == i % 100:
            print(f"{i}/{TRAIN_TIMES} loss=", loss_val)
    print("Train Finished")

    buffer = []
    for i in range(test_batch.count()):
        batch_x, batch_y = test_batch.next()
        batch_x = batch_x.reshape([-1, TIME_STEP, INPUT_SIZE])
        batch_y = batch_y.reshape([-1, TIME_STEP, OUTPUT_SIZE])
        loss_val = sess.run(loss, feed_dict={
            place_x: batch_x,
            place_y: batch_y
        })
        buffer.append(loss_val)
    print("Test loss=", np.array(buffer).mean())
