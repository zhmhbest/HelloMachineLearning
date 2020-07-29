import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from math import ceil
from zhmh.tf.data import BatchGenerator

"""
    —————————————————— 导入数据 ——————————————————
"""
data_all = pd.read_csv('./DATASET/stock.csv')
data_x = data_all[['open', 'close', 'low', 'high', 'volume', 'money', 'change']].to_numpy()
data_y = data_all[['label']].to_numpy()
INPUT_SIZE = data_x.shape[1]
OUTPUT_SIZE = data_y.shape[1]
print(INPUT_SIZE, OUTPUT_SIZE)

"""
    —————————————————— 数据预处理 ——————————————————
"""
# 标准化
std_x = StandardScaler()
data_x = std_x.fit_transform(data_x)
std_y = StandardScaler()
data_y = std_y.fit_transform(data_y)

# 划分数据集
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.25)
TRAIN_SIZE = train_x.shape[0]
TEST_SIZE = test_x.shape[0]
print(TRAIN_SIZE, TEST_SIZE)
# exit()

"""
    —————————————————— 定义网络 ——————————————————
"""
TIME_STEP = 15
RNN_UNIT_SIZE = 10
BATCH_SIZE = 80
SINGLE_GROUP_SIZE = BATCH_SIZE * TIME_STEP
TRAIN_BATCH = BatchGenerator(train_x, train_y, SINGLE_GROUP_SIZE)
TEST_BATCH = BatchGenerator(test_x, test_y, SINGLE_GROUP_SIZE)

place_x = tf.placeholder(tf.float32, shape=[None, TIME_STEP, INPUT_SIZE])
place_y = tf.placeholder(tf.float32, shape=[None, TIME_STEP, OUTPUT_SIZE])
weights = [
    tf.Variable(tf.random_normal([INPUT_SIZE, RNN_UNIT_SIZE])),
    tf.Variable(tf.random_normal([RNN_UNIT_SIZE, OUTPUT_SIZE]))
]
biases = [
    tf.Variable(tf.constant(0.1, shape=[RNN_UNIT_SIZE, ])),
    tf.Variable(tf.constant(0.1, shape=[OUTPUT_SIZE, ]))
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
pred = tf.matmul(
    tf.reshape(output_rnn, [-1, RNN_UNIT_SIZE]),
    weights[1]
) + biases[1]

loss = tf.reduce_mean(
    tf.square(
        tf.reshape(pred, [-1]) -
        tf.reshape(place_y, [-1])
    )
)

"""
    —————————————————— 训练网络 ——————————————————
"""
TRAIN_TIMES = 200 * 50
LEARNING_RATE = 0.0006

train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(TRAIN_TIMES):
        # 每次进行训练的时候，每个batch训练batch_size个样本
        batch_x, batch_y = TRAIN_BATCH.next()
        batch_x = batch_x.reshape([-1, TIME_STEP, INPUT_SIZE])
        batch_y = batch_y.reshape([-1, TIME_STEP, OUTPUT_SIZE])
        # print(batch_x.shape)
        # exit()
        _, loss_val = sess.run(
            [train_op, loss],
            feed_dict={
                place_x: batch_x,
                place_y: batch_y
            }
        )
        if 0 == i % 50:
            print(i, loss_val)
print("Train Finished")

"""
    —————————————————— 预测 ——————————————————
"""
