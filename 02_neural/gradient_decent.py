import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from com.zhmh.tf import \
    generate_input_tensor, \
    generate_relation_data, \
    generate_elu_l2_layers, \
    get_regularized_loss, \
    do_train
from com.zhmh.tf import gpu_first
gpu_first()


"""
    配置参数
"""
SAMPLE_SIZE = 600
TRAINING_TIMES = 6000
LEARNING_RATE = 0.001


"""
    生成随机数据
"""


def data_relation(feature_num, target_num):
    """
    以原点为圆心，半径为1的圆把散点划分成红蓝两部分，并加入随机噪音。
    :return:
    """
    _x_ = np.random.uniform(-1, 1)
    _y_ = np.random.uniform(0, 2)
    return \
        np.random.normal(_x_, 0.1), \
        np.random.normal(_y_, 0.1), \
        (0 if _x_ ** 2 + _y_ ** 2 <= 1 else 1)
# end def


np.random.seed(0)
DATA_X, DATA_Y = generate_relation_data(SAMPLE_SIZE, 2, 1, data_relation)


"""
    定义神经网络
"""
layer_neurons = [5, 4, 3]
input_x, input_y = generate_input_tensor(2, 1, layer_neurons)
calc_y = generate_elu_l2_layers(enter_layer=input_x, layer_neurons=layer_neurons)


"""
    定义损失函数
"""
mse_loss = tf.reduce_sum(tf.pow(input_y - calc_y, 2)) / SAMPLE_SIZE
l2_loss = get_regularized_loss(mse_loss)


"""
    训练
"""


def paint_scatter_data(feature, label, split_line=None, c1='HotPink', c2='DarkCyan'):
    plt.scatter(feature[:, 0], feature[:, 1],
                c=[c1 if _i_ == 0 else c2 for _i_ in label],
                cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white")
    if split_line is not None:
        plt.contour(split_line['x'], split_line['y'], split_line['probs'], levels=[.5], cmap="Greys", vmin=0, vmax=.1)
    plt.show()


def do_sub(sess):
    # 计算分割曲线
    xx, yy = np.mgrid[-1.2:1.2:.01, -0.2:2.2:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = sess.run(calc_y, feed_dict={input_x: grid}).reshape(xx.shape)
    # 画出
    paint_scatter_data(DATA_X, DATA_Y, {'x': xx, 'y': yy, 'probs': probs})


# 欠拟合
do_train(
    tf.train.AdamOptimizer(LEARNING_RATE).minimize(mse_loss),
    500,
    {input_x: DATA_X, input_y: DATA_Y},
    train_after=do_sub
)


# 过拟合
do_train(
    tf.train.AdamOptimizer(LEARNING_RATE).minimize(mse_loss),
    TRAINING_TIMES,
    {input_x: DATA_X, input_y: DATA_Y},
    train_after=do_sub
)


# 加入L2正则化
do_train(
    tf.train.AdamOptimizer(LEARNING_RATE).minimize(l2_loss),
    TRAINING_TIMES,
    {input_x: DATA_X, input_y: DATA_Y},
    train_after=do_sub
)
