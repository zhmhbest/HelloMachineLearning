import os
import tensorflow as tf


def set_log_level(level=1):
    """
    屏蔽通知
    :param level: 0:不屏蔽 | 1:屏蔽通知 | 2:屏蔽警告 | 3:屏蔽错误
    :return:
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(level)


def force_use_cpu():
    """
    强制使用CPU
    :return:
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 隐藏GPU


def gpu_first():
    """
    优先使用GPU
    :return:
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def __is_number(arg):
    return type(arg) is float or type(arg) is int


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
    # 权重初始方法
    w_initializer = \
        (lambda: tf.truncated_normal_initializer(stddev=w_initialize)) if __is_number(w_initialize) else \
        w_initialize
    # print(type(w_initializer()))
    # 偏执初始方法
    b_initializer = \
        (lambda: tf.constant_initializer(b_initialize)) if __is_number(b_initialize) else \
        b_initialize
    # print(type(b_initializer()))

    # 默认ReLu激活
    if build_lambda is None:
        build_lambda = (
            lambda _x, _w, _b, _final:
                tf.matmul(_x, _w) + _b
            if _final else
                tf.nn.relu(tf.matmul(_x, _w) + _b)
        )
    # 构建网络
    x = input_tensor
    for __i in range(1, len(layer_neurons)):
        layer_io = layer_neurons[__i-1], layer_neurons[__i]
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
            with tf.variable_scope('layer' + str(__i), reuse=True):
                weights = tf.get_variable(name='weights')
                biases = tf.get_variable(name='biases')
        # BuildNet
        x = build_lambda(x, weights, biases, __i+1 == len(layer_neurons))
    return x


class TensorBoard:
    def __init__(self, summary_dir="./summary"):
        assert os.path.isdir(os.path.dirname(summary_dir)) is True  # 上级目录存在
        if os.path.exists(summary_dir):
            assert os.path.isdir(summary_dir) is True  # 指定目录不是文件
        else:
            os.makedirs(summary_dir)
        # end if
        self.summary_dir = os.path.abspath(summary_dir)

    def remake(self):
        os.system('RMDIR /S /Q "' + self.summary_dir + '"')
        os.system('MKDIR "' + self.summary_dir + '"')

    def save(self, g):
        tf.summary.FileWriter(self.summary_dir, graph=g)

    def board(self):
        print("TensorBoard may view at:")
        print(" * http://%s:6006/" % os.environ['ComputerName'])
        print(" * http://localhost:6006/")
        print(" * http://127.0.0.1:6006/")
        os.system('tensorboard --logdir="' + self.summary_dir + '"')
