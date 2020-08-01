import tensorflow as tf
from zhmh.data import BatchGenerator

"""
    加载数据
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
"""
from zhmh.dataset.mnist import load_mnist_data

(x_train, y_train, size_train), (x_test, y_test, size_test) = load_mnist_data()
INPUT_SIZE = x_train.shape[1]
OUTPUT_SIZE = y_train.shape[1]

"""
    创建神经网络
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
"""
LAYER_NEURONS = [INPUT_SIZE, 500, OUTPUT_SIZE]
REGULARIZER_COLLECTION = 'losses'
REGULARIZATION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99

place_x = tf.placeholder(tf.float32, shape=(None, INPUT_SIZE))
place_y = tf.placeholder(tf.float32, shape=(None, OUTPUT_SIZE))

l2_regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
global_step = tf.Variable(0, trainable=False)
ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

x = place_x
y = None
for _i_ in range(1, len(LAYER_NEURONS)):
    layer_i, layer_o = LAYER_NEURONS[_i_ - 1], LAYER_NEURONS[_i_]
    with tf.variable_scope('layer' + str(_i_)):
        weights = tf.get_variable(
            name='weights',
            shape=[layer_i, layer_o],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable(
            name='biases',
            shape=[layer_o],
            initializer=tf.constant_initializer(0.001))

        # 正则化收集器
        tf.add_to_collection(REGULARIZER_COLLECTION, l2_regularizer(weights))

        if _i_ != len(LAYER_NEURONS) - 1:
            x = tf.nn.relu(tf.matmul(x, weights) + biases)
        else:
            y = tf.matmul(x, weights) + biases

# 创建影子变量
ema_op = ema.apply(tf.trainable_variables())
x = place_x
y_ema = None
for _i_ in range(1, len(LAYER_NEURONS)):
    layer_i, layer_o = LAYER_NEURONS[_i_ - 1], LAYER_NEURONS[_i_]
    with tf.variable_scope('layer' + str(_i_), reuse=True):
        # 获取已经定义的变量
        weights = tf.get_variable(name='weights')
        biases = tf.get_variable(name='biases')

        if _i_ != len(LAYER_NEURONS) - 1:
            x = tf.nn.relu(tf.matmul(x, ema.average(weights)) + ema.average(biases))
        else:
            y_ema = tf.matmul(x, weights) + biases

"""
    定义损失函数
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
"""
cross_entropy = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(place_y, 1))
)
loss = tf.add_n(tf.get_collection(REGULARIZER_COLLECTION)) + cross_entropy
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(place_y, 1)), tf.float32))
averages_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_ema, 1), tf.argmax(place_y, 1)), tf.float32))

"""
    训练
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
"""
LEARNING_RATE_BASE = 0.8
BATCH_SIZE = 512
LEARNING_RATE_DECAY = 0.99
TRAINING_TIMES = 5000

learning_rate = tf.train.exponential_decay(
    LEARNING_RATE_BASE,
    global_step,
    size_train / BATCH_SIZE, LEARNING_RATE_DECAY,
    staircase=True
)
batch = BatchGenerator(x_train, y_train, BATCH_SIZE, BATCH_SIZE/2)

train_adam = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
with tf.control_dependencies([train_adam, y_ema]):
    train_op = tf.no_op(name='train')

with tf.Session() as sess:
    # 初始化全部变量OP
    tf.global_variables_initializer().run()
    x_batch, y_batch = batch.next()
    for i in range(1, 1 + TRAINING_TIMES):
        sess.run(train_op, feed_dict={
            place_x: x_batch,
            place_y: y_batch
        })
        if i % 1000 == 0:
            va, vl, vg = sess.run([accuracy, loss, global_step], feed_dict={
                place_x: x_train,
                place_y: y_train
            })
            print(f"{vg}、accuracy={va}、loss={vl}")

    common_feed_train = {
        place_x: x_train,
        place_y: y_train
    }
    common_feed_test = {
        place_x: x_test,
        place_y: y_test
    }
    print("Train Accuracy:", accuracy.eval(common_feed_train))
    print("Test Accuracy:", accuracy.eval(common_feed_test))
    print("Train Averages Accuracy:", averages_accuracy.eval(common_feed_train))
    print("Test Averages Accuracy:", averages_accuracy.eval(common_feed_test))
    print(sess.run(
        tf.argmax(y_test[:30], 1)), "Real Number")
    print(sess.run(
        tf.argmax(y[:30], 1), feed_dict=common_feed_test), "Prediction Number")
    print(sess.run(
        tf.argmax(y_ema[:30], 1), feed_dict=common_feed_test), "Prediction Averages Number")
