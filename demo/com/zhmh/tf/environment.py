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
