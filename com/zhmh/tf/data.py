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


def generate_relation_data(row_number, feature_col_number, target_col_number, relation_fn):
    """
    生成关系数据
    :param row_number: 数据条数
    :param feature_col_number:  特征值数量
    :param target_col_number:   目标值数量
    :param relation_fn: 关系函数(feature_col_number, target_col_number)
                        每次返回一组(特征值+目标值)
    :return: (特征数据, 目标数据)

    demo:
        def test(feature_num, target_num):
            # print(r, c)
            _x_ = np.random.uniform(-1, 1)
            _y_ = np.random.uniform(0, 2)
            return _x_, _y_, (0 if _x_ ** 2 + _y_ ** 2 <= 1 else 1)
        X, Y = generate_relation_data(10, 2, 1, test)
        print(X)
        print(Y)
    """
    from com.zhmh.magic import __assert__
    data_col_number = feature_col_number + target_col_number
    dataset = []
    for i in range(row_number):
        one_row = relation_fn(feature_col_number, target_col_number)
        __assert__(
            isinstance(one_row, tuple) and data_col_number == len(one_row),
            "关系函数返回值错误"
        )
        dataset.append(one_row)
    dataset = np.array(dataset)  # 重新封装数据
    return dataset[:, 0:feature_col_number], dataset[:, feature_col_number:data_col_number]


def next_batch(feature, target, index, data_size, batch_size):
    """
    获取下一组batch
    :param feature: 特征数据
    :param target:  目标数据
    :param index:   第几组
    :param data_size:  数据总数
    :param batch_size: 每组个数
    :return:
    """
    bound_l = (index * batch_size) % data_size
    bound_r = min(bound_l + batch_size, data_size)
    return feature[bound_l:bound_r], target[bound_l:bound_r]
