

def load_mnist_data():
    """
        [train-images-idx3-ubyte.gz-训练集特征值](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz)
        [train-labels-idx1-ubyte.gz-训练集目标值](http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz)
        [t10k-images-idx3-ubyte.gz-测试集特征值](http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz)
        [t10k-labels-idx1-ubyte.gz-测试集目标值](http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz)
    :return:
    """
    import os
    import tensorflow as tf
    from tensorflow.examples.tutorials.mnist import input_data

    __log_ver = tf.logging.get_verbosity()
    tf.logging.set_verbosity(tf.logging.ERROR)
    print("Loading mnist data ...")
    mnist = input_data.read_data_sets(os.path.dirname(__file__), one_hot=True)
    tf.logging.set_verbosity(__log_ver)

    # (x_train, y_train, size_train), (x_test, y_test, size_test) = load_mnist_data()
    return (
        mnist.train.images,
        mnist.train.labels,
        mnist.train.num_examples
    ), (
        mnist.test.images,
        mnist.test.labels,
        mnist.test.num_examples
    )
