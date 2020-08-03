import tensorflow as tf
from .common import __init_w_b


def generate_rnn_layer(
        rnn_layers: [int, int, int, int],
        input_tensor,
        w_initialize,
        b_initialize,
        base_cell=tf.nn.rnn_cell.BasicLSTMCell
):
    """
    :param rnn_layers: [input_size, rnn_unit, rnn_deep, output_size]
        rnn_unit: 每个隐层神经元的个数
        rnn_deep: 隐层层数
    :param input_tensor: tf.placeholder(tf.float32, shape=[None, None, INPUT_SIZE])
    :param w_initialize:
    :param b_initialize:
    :param base_cell: BasicLSTMCell | BasicRNNCell
    :return:
    """
    input_size, rnn_unit, rnn_deep, output_size = rnn_layers
    # x = input_tensor
    w_initializer, b_initializer = __init_w_b(w_initialize, b_initialize)
    input_shape = tf.shape(input_tensor)
    batch_size = input_shape[0]
    time_step = input_shape[1]

    # Variable
    with tf.variable_scope('layer_rnn_input'):
        weights_input = tf.get_variable(name='weights', shape=[input_size, rnn_unit], initializer=w_initializer())
        biases_input = tf.get_variable(name='biases', shape=[rnn_unit], initializer=b_initializer())
    with tf.variable_scope('layer_rnn_output'):
        weights_output = tf.get_variable(name='weights', shape=[rnn_unit, output_size], initializer=w_initializer())
        biases_output = tf.get_variable(name='biases', shape=[output_size], initializer=b_initializer())

    # Cell
    generate_one_cell = (lambda: base_cell(rnn_unit))
    cell = tf.nn.rnn_cell.MultiRNNCell([generate_one_cell() for i in range(rnn_deep)])
    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    # 建立网络
    input_reshaped = tf.reshape(input_tensor, [-1, input_size])
    input_weighted = tf.matmul(input_reshaped, weights_input) + biases_input
    rnn_input = tf.reshape(input_weighted, [-1, time_step, rnn_unit])
    rnn_output, final_states = tf.nn.dynamic_rnn(cell, rnn_input, initial_state=init_state, dtype=tf.float32)

    y_pred = tf.matmul(
        tf.reshape(rnn_output, [-1, rnn_unit]),
        weights_output
    ) + biases_output

    return y_pred, final_states
