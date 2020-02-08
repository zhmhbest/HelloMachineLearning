import tensorflow as tf
from com.zhmh.tf import load_mnist_data
from com.zhmh.tf import generate_input_tensor, generate_activation_l2_ema_layers, get_regularized_loss
from com.zhmh.tf import do_train
from com.zhmh.tf import gpu_first
gpu_first()


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
MODEL_SAVE = "./dump/mnist_model"


"""
    加载数据
"""
mnist_data = load_mnist_data()
mnist_data.set_batch_size(BATCH_SIZE)


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
mnist_data.set_input(input_x, input_y)


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
    mnist_data.get_train_size() / BATCH_SIZE, LEARNING_RATE_DECAY,
    staircase=True
)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
with tf.control_dependencies([train_step, averages_op]):
    train_op = tf.no_op(name='train')


def train_what(sess, i):
    global mnist_data
    global train_op, global_step
    global accuracy, loss, y, averages_y

    # xs, ys = mnist.train.next_batch(BATCH_SIZE)
    feed_dict = mnist_data.next_batch(i)
    sess.run(train_op, feed_dict=feed_dict)
    if i % 1000 == 0:
        va, vl, vg = sess.run([accuracy, loss, global_step], feed_dict=feed_dict)
        print(f"{vg}、accuracy={va}、loss={vl}")


def train_after(sess):
    global mnist_data
    global train_op, global_step
    global accuracy, loss, y, averages_y

    print("Train Accuracy:", accuracy.eval(mnist_data.get_train_feed()))
    print("Test Accuracy:", accuracy.eval(mnist_data.get_test_feed()))
    print("Train Averages Accuracy:", averages_accuracy.eval(mnist_data.get_train_feed()))
    print("Test Averages Accuracy:", averages_accuracy.eval(mnist_data.get_test_feed()))
    print(sess.run(
        tf.argmax(mnist_data.get_test_data()['target'][:30], 1)), "Real Number")
    print(sess.run(
        tf.argmax(y[:30], 1), feed_dict=mnist_data.get_test_feed()), "Prediction Number")
    print(sess.run(
        tf.argmax(averages_y[:30], 1), feed_dict=mnist_data.get_test_feed()), "Prediction Averages Number")
    saver.save(sess, MODEL_SAVE, global_step=global_step)


saver = tf.train.Saver()
do_train(TRAINING_TIMES, train_what, train_after=train_after)

