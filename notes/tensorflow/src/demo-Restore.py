import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from zhmh import make_cache
from zhmh.dataset import generate_random_data, BatchGenerator
from zhmh.tf import generate_network
make_cache('./cache/restore')


"""
    生成模拟数据集
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
"""
DATA_SIZE = 256
INPUT_SIZE = 2
OUTPUT_SIZE = 1
np.random.seed(1)  # 固定数据generate_random_data生成的数据
x_data, y_data = generate_random_data(DATA_SIZE, INPUT_SIZE, OUTPUT_SIZE)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25)
# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)
# exit()


def model_train_save(model_location):
    """
        定义网络
        训练网络
        保存网络
        ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    """
    place_x = tf.placeholder(tf.float32, shape=(None, INPUT_SIZE), name='X')
    place_y = tf.placeholder(tf.float32, shape=(None, OUTPUT_SIZE), name='Y')
    y = generate_network([INPUT_SIZE, 3, OUTPUT_SIZE], place_x, 1, 0.001)
    loss = tf.losses.mean_squared_error(place_y, y)
    train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
    batch = BatchGenerator(x_train, y_train, 8)
    train_times = batch.count() * 10
    """
    place_x : Tensor("X:0", shape=(?, 2), dtype=float32)
    place_y : Tensor("Y:0", shape=(?, 1), dtype=float32)
    y       : Tensor("add_1:0", shape=(?, 1), dtype=float32)
    loss    : Tensor("mean_squared_error/value:0", shape=(), dtype=float32)
    """
    # Saver必须在定义完网络之后才能实例化
    saver = tf.train.Saver()

    print("Training")
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(1, 1 + train_times):
            batch_x, batch_y = batch.next()
            sess.run(train_op, feed_dict={
                place_x: batch_x,
                place_y: batch_y
            })
            if i % 200 == 0:
                loss_value = sess.run(loss, feed_dict={place_x: x_train, place_y: y_train})
                print("%d: loss=%g。" % (i, loss_value))
        saver.save(sess, model_location)
        print("Trained")

        # 计算损失值
        loss_val_train = sess.run(loss, feed_dict={place_x: x_train, place_y: y_train})
        loss_val_test = sess.run(loss, feed_dict={place_x: x_test, place_y: y_test})
        print('loss train:', loss_val_train)
        print('loss test:', loss_val_test)


def model_load(model_location):
    """
        模型加载
        ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    """
    print("Trained")

    # 恢复图内Tensor
    saver = tf.train.import_meta_graph(f"{model_location}.meta")
    g = tf.get_default_graph()
    place_x = g.get_tensor_by_name('X:0')
    place_y = g.get_tensor_by_name('Y:0')
    y = g.get_tensor_by_name('add_1:0')
    loss = g.get_tensor_by_name('mean_squared_error/value:0')
    # print(place_x, place_y, y, loss)

    with tf.Session() as sess:
        # 恢复变量的值
        saver.restore(sess, model_location)

        # 计算损失值
        loss_val_train = sess.run(loss, feed_dict={place_x: x_train, place_y: y_train})
        loss_val_test = sess.run(loss, feed_dict={place_x: x_test, place_y: y_test})
        print('loss train:', loss_val_train)
        print('loss test:', loss_val_test)


if __name__ == '__main__':
    model = "./cache/restore/model.ckpt"
    if os.path.exists(f"{model}.meta"):
        model_load(model)
    else:
        model_train_save(model)
