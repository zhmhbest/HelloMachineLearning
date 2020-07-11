import os
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from zhmh.tf.data import DataHolder


def load_mnist_data():
    """
        [train-images-idx3-ubyte.gz-训练集特征值](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz)
        [train-labels-idx1-ubyte.gz-训练集目标值](http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz)
        [t10k-images-idx3-ubyte.gz-测试集特征值](http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz)
        [t10k-labels-idx1-ubyte.gz-测试集目标值](http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz)
    :return:
    """
    log_v = tf.logging.get_verbosity()
    tf.logging.set_verbosity(tf.logging.ERROR)
    print("Loading mnist data ...")
    mnist = input_data.read_data_sets(os.path.dirname(__file__), one_hot=True)
    tf.logging.set_verbosity(log_v)

    return DataHolder({
        'train': {
            'feature':    mnist.train.images,
            'target':     mnist.train.labels,
            'size':       mnist.train.num_examples
        },
        'test': {
            'feature':    mnist.test.images,
            'target':     mnist.test.labels,
            'size':       mnist.test.num_examples
        }
    })
