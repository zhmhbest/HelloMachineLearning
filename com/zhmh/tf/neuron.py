import tensorflow as tf


def generate_input_tensor(feature_number, target_number):
    return \
        tf.placeholder(dtype=tf.float32, shape=(None, feature_number), name='input_x'), \
        tf.placeholder(dtype=tf.float32, shape=(None, target_number), name='input_y')


def generate_wb_layers(
        enter_layer,
        layers_number,
        init_w=None,
        init_b=None,
        next_layer=None
):
    """
    创建新的网络（仅有权重和偏置顶）
    :param {tensor} enter_layer: 输入层
        tf.placeholder(dtype=, shape=(None, feature_number), name=)
    :param {list} layers_number: 每层神经元个数，其中列表第一个元素为输入层特征数量，最后一个元素为输出层数量
    :param init_w: function(current_layer) 如何初始化权重
    :param init_b: function(current_layer) 如何初始化偏置顶
    :param next_layer: function(weight, biases, x) 其它修正
    :return:
    """
    if init_w is None:
        init_w = (lambda current_l: tf.random_normal(shape=current_l, stddev=1))
    if init_b is None:
        init_b = (lambda current_l: tf.constant(value=0.1, shape=[current_l[1]]))
    if next_layer is None:
        next_layer = (lambda current_w, current_b, current_x: current_x)
    # ==================================== #
    x = enter_layer
    current_layer = [-1, -1]
    for _i_ in range(1, len(layers_number)):
        current_layer[0], current_layer[1] = layers_number[_i_ - 1], layers_number[_i_]
        # ------------------------------------ #
        weight = tf.Variable(init_w(current_layer))
        biases = tf.Variable(init_b(current_layer))
        x = tf.matmul(x, weight) + biases
        x = next_layer(weight, biases, x)
        # ------------------------------------ #
    # end for
    return x


def generate_sigmoid_layers(**kwargs):
    y = generate_wb_layers(**kwargs, next_layer=(lambda w, b, x: tf.nn.sigmoid(x)))
    return y


def generate_elu_l2_layers(regularizer_weight=0.003, loss_collection='losses', **kwargs):
    def do_next(w, b, x):
        tf.add_to_collection(loss_collection, l2_regularizer(w))
        return tf.nn.elu(x)

    l2_regularizer = tf.contrib.layers.l2_regularizer(regularizer_weight)
    y = generate_wb_layers(**kwargs, next_layer=do_next)
    return y
