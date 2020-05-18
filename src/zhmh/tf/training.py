import tensorflow as tf
from zhmh.rich import RichPrint


def do_train(
        train_times,
        train_what,

        train_before=None,
        train_after=None
):
    """
    训练
    :param train_times: 训练次数
    :param train_what: function(sess, index)

    :param train_before: function(sess)
    :param train_after: function(sess)
    :return:
    """
    show_every = train_times / 200

    def progress_bar(index):
        """
        打印进度条
        :param index:
        :return:
        """
        nonlocal show_every
        if 0 == index % show_every:
            RichPrint.progress_bar(index / train_times)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        if train_before is not None:
            train_before(sess)

        for i in range(1, 1 + train_times):
            train_what(sess, i)
            progress_bar(i)
        # end for

        if train_after is not None:
            train_after(sess)


def do_simple_train(train_optimizer, feed_dict, **kwargs):
    def train_what(sess, i):
        sess.run(train_optimizer, feed_dict=feed_dict)
    do_train(**kwargs, train_what=train_what)
