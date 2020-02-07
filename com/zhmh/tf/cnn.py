
import tensorflow as tf


def predict_shape(input_w_num, input_h_num, filter_l, filter_s, filter_p):
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
        input_x, deep,
        filter_shape, filter_step,
        pool_shape, pool_step,
        init_w=None, init_b=None
):
    """
    数据卷积
    :param input_x: placeholder(shape=[batch, height, width, channels])
    :param deep: (input_channels, output_channels) （输入深度，输出深度）

    :param filter_shape: (filter_height, filter_width) 过滤器尺寸
    :param filter_step: (step_h, step_w) 过滤器步长

    :param pool_shape: (pool_height, pool_width) 池化尺寸
    :param pool_step: (step_h, step_w) 池化步长

    :param init_w:
    :param init_b:
    :return:
    """
    if isinstance(filter_shape, int):
        filter_shape = (filter_shape, filter_shape)
    if isinstance(filter_step, int):
        filter_step = (filter_step, filter_step)
    if isinstance(pool_shape, int):
        pool_shape = (pool_shape, pool_shape)
    if isinstance(pool_step, int):
        pool_step = (pool_step, pool_step)
    if init_w is None:
        init_w = (lambda: tf.truncated_normal_initializer(stddev=1.0))
    if init_b is None:
        init_b = (lambda: tf.constant_initializer(0))

    weights = tf.get_variable('weights', [filter_shape[0], filter_shape[1], deep[0], deep[1]], initializer=init_w())
    biases = tf.get_variable('biases', [deep[1]], initializer=init_b())

    # 卷积
    # strides: [1, 横向步长, 纵向步长, 1]
    # padding: SAME:全0填充 | VALID
    conv = tf.nn.conv2d(input_x, weights, strides=[1, filter_step[0], filter_step[1], 1], padding='SAME')

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

    return {
        'w': weights,
        'b': biases,
        'y': y
    }
