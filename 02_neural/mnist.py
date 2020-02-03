import tensorflow as tf
from com.zhmh.tf import generate_input_tensor, generate_activation_l2_ema_layers, get_regularized_loss
from com.zhmh.tf import do_train
from com.zhmh.tf import gpu_first
gpu_first()


def load_mnist_data():
    """
        [train-images-idx3-ubyte.gz-训练集特征值](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz)
        [train-labels-idx1-ubyte.gz-训练集目标值](http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz)
        [t10k-images-idx3-ubyte.gz-测试集特征值](http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz)
        [t10k-labels-idx1-ubyte.gz-测试集目标值](http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz)
    :return:
    """
    from tensorflow.examples.tutorials.mnist import input_data
    log_v = tf.logging.get_verbosity()
    tf.logging.set_verbosity(tf.logging.ERROR)
    mnist = input_data.read_data_sets("./dump/MNIST_data/", one_hot=True)
    tf.logging.set_verbosity(log_v)
    # x_train, y_train, x_test, y_test = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    # print('X_train.shape:', x_train.shape)
    # print('X_test.shape:', x_test.shape)
    # print('y_train.shape:', y_train.shape)
    # print('y_test.shape:', y_test.shape)
    # return x_train, y_train, x_test, y_test, mnist
    return mnist


"""
    相关参数
"""
INPUT_NODE = 784
OUTPUT_NODE = 10
HIDDEN_LAYERS = [500]

BATCH_SIZE = 128
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_TIMES = 10000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE = "./dump/MNIST_model/mnist_model"


"""
    加载数据
"""
mnist = load_mnist_data()


"""
    创建神经网络
"""
input_x, input_y = generate_input_tensor(INPUT_NODE, OUTPUT_NODE, HIDDEN_LAYERS)
y, averages_y, averages_op, global_step = generate_activation_l2_ema_layers(
    MOVING_AVERAGE_DECAY,
    REGULARIZATION_RATE,
    enter_layer=input_x, layer_neurons=HIDDEN_LAYERS, activation=tf.nn.relu,
    init_w=(lambda c: tf.truncated_normal_initializer(stddev=0.1))
)


"""
    定义损失函数
"""
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(input_y, 1)))
loss = get_regularized_loss(cross_entropy)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(input_y, 1)), tf.float32))
averages_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(averages_y, 1), tf.argmax(input_y, 1)), tf.float32))


"""
    训练
"""
learning_rate = tf.train.exponential_decay(
    LEARNING_RATE_BASE,
    global_step,
    mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
    staircase=True
)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
with tf.control_dependencies([train_step, averages_op]):
    train_op = tf.no_op(name='train')


def train_what(sess, i):
    global input_x, input_y
    global train_op, global_step
    global accuracy, loss, y, averages_y

    xs, ys = mnist.train.next_batch(BATCH_SIZE)
    sess.run(train_op, feed_dict={input_x: xs, input_y: ys})
    if i % 1000 == 0:
        va, vl, vg = sess.run([accuracy, loss, global_step], feed_dict={input_x: xs, input_y: ys})
        print(f"{vg}、accuracy={va}、loss={vl}")


def train_after(sess):
    global input_x, input_y
    global train_op, global_step
    global accuracy, loss, y, averages_y
    x_train, y_train, x_test, y_test = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    print("Train Accuracy:", accuracy.eval({input_x: x_train, input_y: y_train}))
    print("Test Accuracy:", accuracy.eval({input_x: x_test, input_y: y_test}))
    print("Train Averages Accuracy:", averages_accuracy.eval({input_x: x_train, input_y: y_train}))
    print("Test Averages Accuracy:", averages_accuracy.eval({input_x: x_test, input_y: y_test}))
    print(sess.run(
        tf.argmax(y_test[:30], 1)), "Real Number")
    print(sess.run(
        tf.argmax(y[:30], 1), feed_dict={input_x: x_test, input_y: y_test}), "Prediction Number")
    print(sess.run(
        tf.argmax(averages_y[:30], 1), feed_dict={input_x: x_test, input_y: y_test}), "Prediction Averages Number")
    saver.save(sess, MODEL_SAVE, global_step=global_step)


saver = tf.train.Saver()
do_train(TRAINING_TIMES, train_what, train_after=train_after)

