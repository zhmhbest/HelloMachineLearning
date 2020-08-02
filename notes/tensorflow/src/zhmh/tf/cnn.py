import tensorflow as tf
from .common import __init_w_b


def get_filtered_size(input_w_num, input_h_num, filter_l, filter_s, filter_p):
    """
    过滤后的输出大小
    :param input_w_num: 输入宽度
    :param input_h_num: 输入高度
    :param filter_l: 过滤器边长
    :param filter_s: 过滤器步长
    :param filter_p: 过滤器边缘
    :return:
    """
    constant_1 = 2 * filter_p - filter_l
    constant_2 = filter_s + 1
    output_w = (input_w_num + constant_1) / constant_2
    output_h = (input_h_num + constant_1) / constant_2
    return output_w, output_h


def generate_one_conv(
        input_tensor,
        deep: (int, int),
        filter_shape: (int, int),
        filter_step: (int, int),
        pool_shape: (int, int),
        pool_step: (int, int),
        w_initialize,
        b_initialize
):
    """
    数据卷积
    :param input_tensor: tf.placeholder(shape=[batch, height, width, channels])
    :param deep: (input_channels, output_channels) （输入深度，输出深度）

    :param filter_shape: (filter_height, filter_width) 过滤器尺寸
    :param filter_step: (step_h, step_w) 过滤器步长

    :param pool_shape: (pool_height, pool_width) 池化尺寸
    :param pool_step: (step_h, step_w) 池化步长

    :param w_initialize:
    :param b_initialize:
    :return:
    """
    w_initializer, b_initializer = __init_w_b(w_initialize, b_initialize)

    weights = tf.get_variable(
        'weights',
        [filter_shape[0], filter_shape[1], deep[0], deep[1]],
        initializer=w_initializer())
    biases = tf.get_variable(
        'biases',
        [deep[1]],
        initializer=b_initializer())

    # 卷积
    # strides: [1, 横向步长, 纵向步长, 1]
    # padding: SAME:全0填充 | VALID
    conv = tf.nn.conv2d(
        input_tensor,
        weights,
        strides=[1, filter_step[0], filter_step[1], 1],
        padding='SAME'
    )

    y = tf.nn.bias_add(conv, biases)
    y = tf.nn.relu(y)

    # 池化
    # ksize: [1, 宽, 高, 1]
    y = tf.nn.avg_pool(
        y,
        ksize=[1, pool_shape[0], pool_shape[1], 1],
        strides=[1, pool_step[0], pool_step[1], 1],
        padding='SAME'
    )

    return y
