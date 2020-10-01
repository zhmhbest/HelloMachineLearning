import numpy as np
from sklearn.preprocessing import OneHotEncoder     # 独热码
from sklearn.preprocessing import MinMaxScaler      # 归一化
from sklearn.preprocessing import StandardScaler    # 标准化
from sklearn.preprocessing import Normalizer        # 正则化


def pp_one_hot(test_data):
    print("【OneHotEncoder】")
    encoder = OneHotEncoder()
    value = encoder.fit_transform(test_data).toarray()
    print(value)
    # 逆变
    origin = encoder.inverse_transform(value)
    print(origin)


def pp_min_max(test_data):
    print("【MinMaxScaler】")
    encoder = MinMaxScaler()
    # 按列
    value = encoder.fit_transform(test_data)
    print(value)
    # 逆变
    origin = encoder.inverse_transform(value)
    print(origin)


def pp_std(test_data):
    print("【StandardScaler】")
    encoder = StandardScaler()
    # 按列
    value = encoder.fit_transform(test_data)
    print(value)
    # 逆变
    origin = encoder.inverse_transform(value)
    print(origin)


def pp_norm(test_data):
    def norm1(_x):
        _norms = np.sum(np.abs(_x), axis=1)[:, np.newaxis]
        return _x / _norms

    def norm2(_x):
        _norms = np.sqrt(np.sum((np.power(_x, 2)), axis=1))[:, np.newaxis]
        return _x / _norms

    print("【Normalizer】")
    encoder_l1 = Normalizer(norm='l1')
    encoder_l2 = Normalizer(norm='l2')
    # 按列
    print(encoder_l1.fit_transform(test_data))
    print(norm1(test_data))

    print(encoder_l2.fit_transform(test_data))
    print(norm2(test_data))


if __name__ == '__main__':
    pp_one_hot(
        np.array(["北京", "上海", "深圳"]).reshape(-1, 1)
    )
    print("="*32)

    data = [
        [1, -1, 3],
        [2, 4, 2],
        [4, 6, -1]
    ]

    pp_min_max(data)
    print("="*32)

    pp_std(data)
    print("="*32)

    pp_norm(data)
