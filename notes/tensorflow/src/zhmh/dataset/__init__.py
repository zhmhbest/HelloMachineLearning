import numpy as np


class BatchGenerator:
    def __init__(self, x_train, y_train, batch_size, step_size=1):
        from zhmh.magic import __assert__
        __assert__(x_train.shape[0] == y_train.shape[0], 'x_train dose not match y_train.')
        self.data_size = x_train.shape[0]
        self.batch_size = int(batch_size)
        __assert__(self.batch_size > 0, 'batch_size <= 0')
        self.step_size = int(step_size)
        __assert__(self.step_size > 0, 'step_size <= 0')
        self.X = x_train
        self.Y = y_train
        self.index = 0

    def count(self):
        return ((self.data_size - self.batch_size) // self.step_size) + 1

    def next(self):
        index = self.index if (self.data_size - self.index) > self.batch_size else 0
        d_x = self.X[index: index + self.batch_size]
        d_y = self.Y[index: index + self.batch_size]
        self.index = index + self.step_size
        return d_x, d_y


def generate_random_data(row_number, feature_col_number, target_col_number=0):
    """
    生成随机数据
    :param row_number: 数据条数
    :param feature_col_number:  特征值数量
    :param target_col_number:   目标值数量
    :return: (特征数据, 目标数据)
    """
    data_col_number = feature_col_number + target_col_number
    dataset = np.random.rand(row_number * data_col_number).reshape(row_number, data_col_number)
    return dataset[:, 0:feature_col_number], dataset[:, feature_col_number:data_col_number]
