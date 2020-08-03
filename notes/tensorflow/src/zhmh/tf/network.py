import tensorflow as tf
from .common import __init_w_b


def get_default_build_lambda(activation_function=tf.nn.relu):
    build_lambda = (
        lambda _x, _w, _b, _final:
        tf.matmul(_x, _w) + _b
        if _final else
        activation_function(tf.matmul(_x, _w) + _b)
    )
    return build_lambda


def get_l2_build_lambda(reg_weight: float, reg_collection: str = 'losses', activation_function=tf.nn.relu):
    from tensorflow.contrib.layers import l2_regularizer
    # tf.contrib.layers.l2_regularizer
    l2_regularizer = l2_regularizer(reg_weight)

    def build_network(x, w, b, is_final):
        nonlocal reg_collection, activation_function
        tf.add_to_collection(reg_collection, l2_regularizer(w))
        if is_final:
            return tf.matmul(x, w) + b
        else:
            return activation_function(tf.matmul(x, w) + b)

    return build_network


def get_ema_build_lambda(global_step, decay, activation_function=tf.nn.relu):
    ema = tf.train.ExponentialMovingAverage(decay, global_step)
    ema_op = ema.apply(tf.trainable_variables())  # 要被训练

    def build_network(x, w, b, is_final):
        nonlocal ema, activation_function
        if is_final:
            return tf.matmul(x, ema.average(w)) + ema.average(b)
        else:
            return activation_function(tf.matmul(x, ema.average(w)) + ema.average(b))

    return build_network, ema_op


def generate_network(
        layer_neurons: [int, ...],
        input_tensor: tf.placeholder,
        w_initialize,
        b_initialize,
        build_lambda=None,
        var_reuse=None
):
    """
    构建全链接神经网络
    :param layer_neurons: [INPUT_SIZE, ..., OUTPUT_SIZE]
    :param input_tensor : tf.placeholder(tf.float32, shape=(None, INPUT_SIZE))
    :param w_initialize: num or (lambda: tf.truncated_normal_initializer(stddev=?))
    :param b_initialize: num or (lambda: tf.constant_initializer(0.001))
    :param build_lambda: (lambda x, w, b, is_final: ...)
    :param var_reuse: 已声明的变量
    :return:
    """
    w_initializer, b_initializer = __init_w_b(w_initialize, b_initialize)

    # 默认构造方法
    if build_lambda is None:
        build_lambda = get_default_build_lambda(tf.nn.relu)

    # 构建网络
    x = input_tensor
    for __i in range(1, len(layer_neurons)):
        layer_io = layer_neurons[__i - 1], layer_neurons[__i]
        if var_reuse is None:
            with tf.variable_scope('layer' + str(__i)):
                weights = tf.get_variable(
                    name='weights',
                    shape=layer_io,
                    initializer=w_initializer())
                biases = tf.get_variable(
                    name='biases',
                    shape=layer_io[1],
                    initializer=b_initializer())
        else:
            # reuse=tf.AUTO_REUSE
            with tf.variable_scope('layer' + str(__i), reuse=True):
                weights = tf.get_variable(name='weights')
                biases = tf.get_variable(name='biases')
        # BuildNet
        x = build_lambda(x, weights, biases, __i + 1 == len(layer_neurons))
    return x
