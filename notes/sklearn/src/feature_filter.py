import numpy as np
# from sklearn.preprocessing import Imputer as SimpleImputer  # 缺失值（低版本scikit-learn使用）
from sklearn.impute import SimpleImputer                      # 缺失值（高版本scikit-learn使用）
from sklearn.feature_selection import VarianceThreshold       # 删除低方差特征
from sklearn.decomposition import PCA                         # 主成分分析数据降维


def filter_nan():
    print("【填补NaN】")
    test_data = [
        [1, 2],
        [np.nan, 3],
        [7, 6]
    ]
    # strategy = mean | median | most_frequent | constant（需要指定fill_value）
    print("* 以平均值填补（按列）")
    print(
        SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(test_data)
    )
    print("* 以固定值填补（按列）")
    print(
        SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-1).fit_transform(test_data)
    )


def filter_threshold():
    print("【删除低方差特征】")
    test_data = np.array([
        [0, 2, 0, 3],
        [0, 1, 4, 3],
        [0, 1, 1, 3]
    ])
    print("* 原始数据")
    print(test_data)
    print("* 方差（按列）")
    print(np.round(np.std(test_data, axis=0), 2))
    print("* 删除（按列）")
    print(
        VarianceThreshold().fit_transform(test_data)
    )


def filter_pca():
    """
    Y=PX
    P = [
      1/sqrt(2),  1/sqrt(2),
      -1/sqrt(2), 1/sqrt(2)
    ]
    --------------------------------
        n_components:
            小数: 保留比例
            整数: 保留下来的特征数量
            字符串: 指定解析方法
    """
    print("【PCA】")
    test_data = [
        [2, 8, 4, 5],
        [6, 3, 0, 8],
        [5, 4, 9, 1]
    ]
    print(
        PCA(n_components=0.9).fit_transform(test_data)
    )


if __name__ == '__main__':
    filter_nan()
    print("="*32)

    filter_threshold()
    print("="*32)

    filter_pca()
