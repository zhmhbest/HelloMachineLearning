import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from zhmh.dataset import BatchGenerator


def get_linear_data(size, slope, noise_intensity, is_plot=False):
    """
    生成模拟数据集 x → y
    :param size:            数据量
    :param slope:           斜率
    :param noise_intensity: 噪点强度
    :param is_plot:         显示数据
    :return:
    """
    _x = np.arange(size)
    _noise = np.random.uniform(-1, 1, size) * noise_intensity
    _y = slope * (_x + _noise)
    if is_plot:
        plt.plot(_x, (slope * _x), label='Origin', linestyle='-.')
    return _x.reshape(-1, 1), _y.reshape(-1, 1)


# 【加载数据】
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
DATA_SIZE = 512
x_train, y_train = get_linear_data(DATA_SIZE, 3, 50, True)
INPUT_SIZE = x_train.shape[-1]
OUTPUT_SIZE = y_train.shape[-1]
print(x_train.shape, y_train.shape)
print(INPUT_SIZE, OUTPUT_SIZE)


# 【构建输入输出层】
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
x_input = tf.placeholder(tf.float32, shape=(None, INPUT_SIZE))
y_input = tf.placeholder(tf.float32, shape=(None, OUTPUT_SIZE))


# 【构建权重和偏置】
# y = wx+b
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
w = tf.Variable(tf.random_normal([INPUT_SIZE, OUTPUT_SIZE], stddev=1))
b = tf.Variable(tf.zeros([OUTPUT_SIZE]))
y = tf.matmul(x_input, w) + b


# 【构建代价函数】
# loss = Σ(y - y_input)^2
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
loss = tf.losses.mean_squared_error(y_input, y)


# 【构建优化器】
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
LEARNING_RATE = 0.01
train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)


# 【训练】
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
BATCH_SIZE = 32
TRAIN_TIMES = 5000
batch = BatchGenerator(x_train, y_train, BATCH_SIZE)

with tf.Session() as sess:
    # 初始化全部变量
    tf.global_variables_initializer().run()

    # 未经训练的参数取值
    w_value = sess.run(w)
    b_value = sess.run(b)
    print(f"训练前：w={w_value}, b={b_value}")

    # 训练
    for i in range(1, 1 + TRAIN_TIMES):
        batch_x, batch_y = batch.next()
        sess.run(train_op, feed_dict={
            x_input: batch_x,
            y_input: batch_y
        })
        if i % 100 == 0:
            loss_value = sess.run(loss, feed_dict={x_input: x_train, y_input: y_train})
            print(f"\t训练{i}次后，损失为{loss_value}")

    # 训练后的参数取值
    w_value = sess.run(w)
    b_value = sess.run(b)
    print(f"训练后：w={w_value}, b={b_value}")


# 【展示结果】
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
y_pred = np.matmul(x_train, w_value) + b_value
plt.scatter(x_train, y_train, label='Real Dataset', c='pink', s=10)
plt.plot(y_pred, label='Fitting', linestyle='-.')
plt.legend()
plt.grid()
plt.show()
