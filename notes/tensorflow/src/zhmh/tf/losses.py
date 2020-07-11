import tensorflow as tf


def cross_entropy(labels, predictions, epsilon=1e-7):
    return -(
            labels * tf.log(tf.clip_by_value(predictions, epsilon, 1.0))
            +
            (1 - labels) * tf.log(tf.clip_by_value(1 - predictions, epsilon, 1.0))
    )