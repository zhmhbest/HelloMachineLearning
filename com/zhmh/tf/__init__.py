
from .board import TensorBoard
from .environment import gpu_first, set_log_level

from .data import \
    generate_random_data, \
    generate_relation_data, \
    generate_random_one_rgb_picture, \
    generate_random_rgb_pictures, \
    show_rgb_picture, \
    next_batch, \
    DataHolder

from .neuron import \
    generate_input_tensor, \
    generate_wb_layers, \
    \
    get_regularized_loss, \
    generate_activation_layers, \
    generate_activation_l2_layers, \
    generate_activation_l2_ema_layers

from .training import \
    do_train, \
    do_simple_train

from .cnn import \
    predict_shape, \
    generate_one_conv

from .mnist import load_mnist_data
