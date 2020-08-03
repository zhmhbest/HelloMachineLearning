import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from zhmh import make_cache
from zhmh.dataset import generate_random_data, BatchGenerator
from zhmh.tf import generate_network, get_ema_build_lambda
make_cache('./cache/restore')


"""
    生成模拟数据集
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
"""
DATA_SIZE = 256
INPUT_SIZE = 2
OUTPUT_SIZE = 1
np.random.seed(1)  # 固定generate_random_data生成的数据
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
    # 定义网络
    place_x = tf.placeholder(tf.float32, shape=(None, INPUT_SIZE), name='X')
    place_y = tf.placeholder(tf.float32, shape=(None, OUTPUT_SIZE), name='Y')
    '''
        place_x : Tensor("X:0", shape=(?, 2), dtype=float32)
        place_y : Tensor("Y:0", shape=(?, 1), dtype=float32)
    '''
    y = generate_network([INPUT_SIZE, 3, OUTPUT_SIZE], place_x, 1, 0.001)
    loss = tf.losses.mean_squared_error(place_y, y)
    '''
        y       : Tensor("add_1:0", shape=(?, 1), dtype=float32)
        loss    : Tensor("mean_squared_error/value:0", shape=(), dtype=float32)
    '''

    # 定义EMA网络
    global_step = tf.Variable(0, trainable=False)
    build_ema, ema_op = get_ema_build_lambda(global_step, 0.99, tf.nn.relu)
    y_ema = generate_network([INPUT_SIZE, 3, OUTPUT_SIZE], place_x, 1, 0.001, build_ema, True)
    loss_ema = tf.losses.mean_squared_error(place_y, y_ema)
    '''
        y_ema   : Tensor("add_3:0", shape=(?, 1), dtype=float32)
        loss_ema: Tensor("mean_squared_error_1/value:0", shape=(), dtype=float32)
    '''

    # 定义优化器
    train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
    train_ops = [train_op, ema_op]

    # Saver必须在定义完网络之后才能实例化
    saver = tf.train.Saver()

    print("Training")
    batch = BatchGenerator(x_train, y_train, 8)
    train_times = batch.count() * 10
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(1, 1 + train_times):
            batch_x, batch_y = batch.next()
            sess.run(train_ops, feed_dict={
                place_x: batch_x,
                place_y: batch_y
            })
            if i % 200 == 0:
                loss_value = sess.run(loss, feed_dict={place_x: x_train, place_y: y_train})
                loss_ema_value = sess.run(loss_ema, feed_dict={place_x: x_train, place_y: y_train})
                print("%d: loss=%g, %g" % (i, loss_value, loss_ema_value))
        print("Trained")
        saver.save(sess, model_location)

        # 计算损失值
        train_feed = {place_x: x_train, place_y: y_train}
        test_feed = {place_x: x_test, place_y: y_test}
        print('loss train:', sess.run(loss, feed_dict=train_feed))
        print('loss test:', sess.run(loss, feed_dict=test_feed))
        print('loss ema train:', sess.run(loss_ema, feed_dict=train_feed))
        print('loss ema test:', sess.run(loss_ema, feed_dict=test_feed))


def model_load(model_location):
    """
        恢复Tensor
        恢复变量值
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
    y_ema = g.get_tensor_by_name('add_3:0')
    loss_ema = g.get_tensor_by_name('mean_squared_error_1/value:0')

    with tf.Session() as sess:
        # 恢复变量的值
        saver.restore(sess, model_location)

        # 计算损失值
        train_feed = {place_x: x_train, place_y: y_train}
        test_feed = {place_x: x_test, place_y: y_test}
        print('loss train:', sess.run(loss, feed_dict=train_feed))
        print('loss test:', sess.run(loss, feed_dict=test_feed))
        print('loss ema train:', sess.run(loss_ema, feed_dict=train_feed))
        print('loss ema test:', sess.run(loss_ema, feed_dict=test_feed))


if __name__ == '__main__':
    model = "./cache/restore/model.ckpt"
    if os.path.exists(f"{model}.meta"):
        model_load(model)
    else:
        model_train_save(model)
