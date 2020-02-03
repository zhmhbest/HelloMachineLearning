import tensorflow as tf
from com.zhmh.tf import generate_input_tensor

"""
    加载数据
"""


def load_mnist_data(data_location='./dump/MNIST_data'):
    import os
    from tensorflow.examples.tutorials.mnist import input_data
    # 屏蔽弃用信息
    saved_v = tf.logging.get_verbosity()
    tf.logging.set_verbosity(tf.logging.ERROR)

    def try_download_mnist_data(location, name):
        import urllib.request
        filename = f'{location}/{name}'
        if not os.path.exists(filename):
            print(f'Downloading mnist [{name}]')
            urllib.request.urlretrieve(f'http://yann.lecun.com/exdb/mnist/{name}', filename)
    # end def
    if not os.path.exists(data_location):
        os.mkdir(data_location)
    # end if
    try_download_mnist_data(data_location, 'train-images-idx3-ubyte.gz')
    try_download_mnist_data(data_location, 'train-labels-idx1-ubyte.gz')
    try_download_mnist_data(data_location, 't10k-images-idx3-ubyte.gz')
    try_download_mnist_data(data_location, 't10k-labels-idx1-ubyte.gz')
    data = input_data.read_data_sets(data_location)
    # 恢复原警告等级
    tf.logging.set_verbosity(saved_v)
    return data


mnist = load_mnist_data()


"""
    相关参数
"""
BATCH_SIZE = 100  # 每次batch打包的样本个数


LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

global_step = tf.Variable(0, trainable=False)
