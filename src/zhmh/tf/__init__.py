import os


def set_log_level(level=1):
    """
    屏蔽通知
    :param level: 0:不屏蔽 | 1:屏蔽通知 | 2:屏蔽警告 | 3:屏蔽错误
    :return:
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(level)


def force_use_cpu():
    """
    强制使用CPU
    :return:
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 隐藏GPU


def gpu_first():
    """
    优先使用GPU
    :return:
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


class TensorBoard:
    def __init__(self, summary_dir="./summary"):
        assert os.path.isdir(os.path.dirname(summary_dir)) is True  # 上级目录存在
        if os.path.exists(summary_dir):
            assert os.path.isdir(summary_dir) is True  # 指定目录不是文件
        else:
            os.makedirs(summary_dir)
        # end if
        self.summary_dir = os.path.abspath(summary_dir)

    def remake(self):
        os.system('RMDIR /S /Q "' + self.summary_dir + '"')
        os.system('MKDIR "' + self.summary_dir + '"')

    def save(self, g):
        import tensorflow as tf
        tf.summary.FileWriter(self.summary_dir, graph=g)

    def board(self):
        print("TensorBoard may view at:")
        print(" * http://%s:6006/" % os.environ['ComputerName'])
        print(" * http://localhost:6006/")
        print(" * http://127.0.0.1:6006/")
        os.system('tensorboard --logdir="' + self.summary_dir + '"')
