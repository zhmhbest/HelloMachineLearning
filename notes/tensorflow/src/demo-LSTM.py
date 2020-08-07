import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from zhmh.dataset import BatchGenerator
from zhmh.dataset.stock_sh1 import load_stock_sh000001_data
from zhmh.tf import generate_rnn_layer
from zhmh import make_cache

make_cache("./cache/lstm")
MODEL_LOCATION = "./cache/lstm/model.ckpt"

"""
    导入数据
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
"""
x_data, y_data = load_stock_sh000001_data()
INPUT_SIZE = x_data.shape[1]
OUTPUT_SIZE = y_data.shape[1]
print(INPUT_SIZE, OUTPUT_SIZE)
# # 数据展示
# plt.figure()
# plt.plot(list(range(len(y_data))), y_data)
# plt.show()

"""
    数据预处理
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
"""

# 标准化
std_x, std_y = StandardScaler(), StandardScaler()
x_data = std_x.fit_transform(x_data)
y_data = std_y.fit_transform(y_data)

# 划分数据集（不允许打乱）
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, shuffle=False)
TRAIN_SIZE = x_train.shape[0]
TEST_SIZE = x_test.shape[0]
print(TRAIN_SIZE, TEST_SIZE)

# # 数据展示
# plt.figure()
# plt.plot(list(range(len(y_train))),
#          std_y.inverse_transform(y_train))
# plt.plot(list(len(y_train) + i for i in range(len(y_test))),
#          std_y.inverse_transform(y_test))
# plt.show()

if not os.path.exists(f"{MODEL_LOCATION}.meta"):
    """
        定义网络
        ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    """
    RNN_UNIT = 10  # 隐层神经元的个数
    RNN_DEEP = 2  # 隐层层数

    place_x = tf.placeholder(tf.float32, shape=[None, None, INPUT_SIZE], name='X')
    place_y = tf.placeholder(tf.float32, shape=[None, None, OUTPUT_SIZE], name='Y')
    y_pred, final_states = generate_rnn_layer([INPUT_SIZE, RNN_UNIT, RNN_DEEP, OUTPUT_SIZE], place_x, 1, 0.01)
    loss = tf.losses.mean_squared_error(tf.reshape(place_y, [-1]), tf.reshape(y_pred, [-1]))
    '''
        y_pred: Tensor("add_1:0", shape=(?, 1), dtype=float32)
        loss  : Tensor("mean_squared_error/value:0", shape=(), dtype=float32)
    '''
    LEARNING_RATE = 0.0006
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    """
        训练
        ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    """
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)

    BATCH_SIZE = 60
    TIME_STEP = 20
    train_batch = BatchGenerator(x_train, y_train, BATCH_SIZE * TIME_STEP, TIME_STEP/2)
    TRAIN_TIMES = train_batch.count() * 15

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(1, TRAIN_TIMES + 1):
            x_batch, y_batch = train_batch.next()
            x_batch = x_batch.reshape([-1, TIME_STEP, INPUT_SIZE])
            y_batch = y_batch.reshape([-1, TIME_STEP, OUTPUT_SIZE])
            sess.run(train_op, feed_dict={
                place_x: x_batch,
                place_y: y_batch
            })
            if 0 == i % 100:
                loss_val = sess.run(loss, feed_dict={
                    place_x: x_batch,
                    place_y: y_batch
                })
                print(f"{round(i / TRAIN_TIMES, 3)}\t loss = {loss_val}")
        print("Trained")

        saver.save(sess, MODEL_LOCATION)

        loss_train_val = sess.run(loss, feed_dict={
            place_x: x_train.reshape([1, -1, INPUT_SIZE]),
            place_y: y_train.reshape([1, -1, OUTPUT_SIZE])
        })
        loss_test_val = sess.run(loss, feed_dict={
            place_x: x_test.reshape([1, -1, INPUT_SIZE]),
            place_y: y_test.reshape([1, -1, OUTPUT_SIZE])
        })
        print('loss train:', loss_train_val)
        print('loss test:', loss_test_val)

else:
    """
        预测
        ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    """
    # 恢复图内Tensor
    saver = tf.train.import_meta_graph(f"{MODEL_LOCATION}.meta")
    g = tf.get_default_graph()
    place_x = g.get_tensor_by_name('X:0')
    place_y = g.get_tensor_by_name('Y:0')
    y_pred = g.get_tensor_by_name('add_1:0')
    loss = g.get_tensor_by_name('mean_squared_error/value:0')

    with tf.Session() as sess:
        saver.restore(sess, MODEL_LOCATION)

        # train
        train_pred = []
        for index in range(len(x_train) - 1):
            prob = sess.run(y_pred, feed_dict={
                place_x: np.array([x_train[index]]).reshape([1, -1, INPUT_SIZE])
            })
            train_pred.extend(prob.reshape([-1]))

        train_pred = std_y.inverse_transform(train_pred)
        train_true = std_y.inverse_transform(y_train)

        # plt.figure()
        # plt.plot(list(range(len(train_true))), train_true, color='r')
        # plt.plot(list(range(len(train_pred))), train_pred, color='b')
        # plt.show()

        # test
        test_pred = []
        for index in range(len(x_test) - 1):
            prob = sess.run(y_pred, feed_dict={
                place_x: np.array([x_test[index]]).reshape([1, -1, INPUT_SIZE])
            })
            test_pred.extend(prob.reshape([-1]))

        test_pred = std_y.inverse_transform(test_pred)
        test_true = std_y.inverse_transform(y_test)

        # plt.figure()
        # plt.plot(list(range(len(test_true))), test_true, color='r')
        # plt.plot(list(range(len(test_pred))), test_pred, color='b')
        # plt.show()

        color_true_train = 'Gold'
        color_true_test = 'Gold'
        color_pred_train = '#FF69B4'
        color_pred_test = '#AA69B4'

        plt.figure(figsize=(36, 8), dpi=300)
        plt.plot(list(range(len(train_true))), train_true, color=color_true_train)
        plt.plot(list(range(len(train_pred))), train_pred, color=color_pred_train)
        plt.plot(list(len(train_true) + i for i in range(len(test_true))), test_true, color=color_true_test)
        plt.plot(list(len(train_pred) + i for i in range(len(test_pred))), test_pred, color=color_pred_test)
        plt.show()
