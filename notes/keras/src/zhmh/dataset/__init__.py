import numpy as np


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
