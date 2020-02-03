
from .board import TensorBoard
from .environment import gpu_first, set_log_level
from .data import generate_random_data, generate_relation_data, next_batch
from .neuron import \
    generate_input_tensor, \
    generate_wb_layers, \
    \
    get_regularized_loss, \
    generate_activation_layers, \
    generate_activation_l2_layers, \
    generate_activation_l2_ema_layers
from .training import do_train, do_simple_train
