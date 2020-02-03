import tensorflow as tf


class RichPrint:
    CONTROL = {
        'DEFAULT': 0,
        'BOLD': 1,
        'UNDERLINE': 4,
        'FLASH': 5,     # 无效
        'SWAP': 7,
        'HIDE': 8,      # 无效
    }
    COLORS = {
        'WHITE':    0, 'w': 0,
        'RED':      1, 'r': 1,
        'GREEN':    2, 'g': 2,
        'YELLOW':   3, 'y': 3,
        'BLUE':     4, 'b': 4,
        'PURPLE':   5, 'p': 5,
        'CYAN':     6, 'c': 6,
        'SHALLOW':  7, 's': 7,
    }

    @staticmethod
    def print_head(style):
        try:
            style[0] = str(RichPrint.CONTROL[style[0]])
        except KeyError:
            style[0] = ''
        # end try
        try:
            style[1] = str(30 + RichPrint.COLORS[style[1]])
        except KeyError:
            style[1] = ''
        # end try
        try:
            style[2] = str(40 + RichPrint.COLORS[style[2]])
        except KeyError:
            style[2] = ''
        # end try
        print(''.join(['\033[',
                       style[0], ';',
                       style[1], ';',
                       style[2],
                       'm']), end='')

    @staticmethod
    def print_tail():
        print('\033[0m', end='')

    @staticmethod
    def p(content, foreground=None, background=None, control=None):
        """
        彩色不换行打印
        :param content: 内容
        :param foreground: 前景色
        :param background: 背景色
        :param control: 其它效果
        :return: None
        """
        RichPrint.print_head([control, foreground, background])
        print(content, end='')
        RichPrint.print_tail()

    @staticmethod
    def pl(content, foreground=None, background=None, control=None):
        """
        彩色换行打印
        :param content: 内容
        :param foreground: 前景色
        :param background: 背景色
        :param control: 其它效果
        :return:
        """
        RichPrint.p(content, foreground, background, control)
        print()

    @staticmethod
    def progress_bar(rate, length=32, single_char1='=', single_char2='.'):
        num_part_1 = int((rate + 0.0001) * length)
        num_part_2 = length - num_part_1
        print('[', end='')
        print(single_char1 * num_part_1, end='')
        print('>', end='')
        print(single_char2 * num_part_2, end='')
        print(']', end=' ')
        # print(rate, end=' ')
        print("%.2f%%" % (rate * 100))


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

