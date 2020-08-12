import numpy as np
import pandas as pd
import math
# 数据集
from keras.datasets import cifar10
# One-Hot
from keras.utils.np_utils import to_categorical
from keras.optimizers import adam
from keras.callbacks import EarlyStopping
from keras.activations import relu, softmax
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy

from keras.models import Sequential, load_model, model_from_json
from keras.constraints import maxnorm
from keras.layers.core import (Dense, Dropout, Activation)
"""
    Dense           : 用于创建密集神经元
    Dropout         : 用于正则化
    Activation      : 用于创建激活层
"""
from keras.layers import (Flatten, Conv2D, MaxPooling2D)
"""
    Flatten         : 多维的输入一维化，想让与reshape((-1, ))
    Conv2D          : 卷积层
    MaxPooling2D    : 池化层
"""


def make_dump(dump_path):
    """
    创建缓存目录
    """
    import os
    if os.path.exists(dump_path):
        if not os.path.isdir(dump_path):
            print(dump_path, "is not a valid path.")
            exit(1)
    else:
        os.mkdir(dump_path)


make_dump('./dump')


# 【数据载入】
# ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■
"""
    cifar10 是一个包含阿拉伯数字图片的数据集
        数据类型: ndarray
        训练数目: 50000
        测试数目: 10000
        图片尺寸: 32 × 32 × 3 = 3072
        输出结果: len(0,1,2,3,4,5,6,7,8,9) = 10
"""
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
INPUT_SHAPE = (lambda x: (x.pop(0), tuple(x)))(list(x_train.shape))[1]


# 【数据预处理】
# ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■
def gaussian_distribution(d):
    """
    高斯分布
    """
    return (d - np.mean(d)) / np.std(d)


""""
    图像在采集、传输和转换过程中都容易受环境的影响，这在图像中就表现为噪声，
    这些噪声会致使图像质量降低或者干扰我们提取原本想要的图像信息，
    所以需要通过滤波技术来去除这些图像中的噪声干扰。
"""
X_train = gaussian_distribution(x_train)
X_test = gaussian_distribution(x_test)
# print(x_train[0:1])
# print(X_train[0:1])

# One-Hot 编码
Y_train = to_categorical(y_train)
Y_test = to_categorical(y_test)
NUM_CLASSES = Y_train.shape[1]
# print(y_train[0:3])
# print(Y_train[0:3])
# print(Y_train.shape, Y_test.shape)
# print(NUM_CLASSES)


# 【模型定义】
# ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■
model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=INPUT_SHAPE, padding='same', activation=relu, kernel_constraint=maxnorm(3)))
model.add(Conv2D(16, (3, 3), activation=relu, padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation=relu, kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(32, activation=relu, kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(10, activation=softmax))


# 【模型编译】
# ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■
# 定义：损失函数、优化器
model.compile(
    loss=categorical_crossentropy,
    optimizer=adam(lr=0.01),
    metrics=[categorical_accuracy]
)


# 【模型训练】
# ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■
model.fit(
    X_train, Y_train,
    batch_size=1000,                    # 每组数量
    epochs=8,                           # 循环次数
    validation_data=(X_test, Y_test),
    callbacks=[
        EarlyStopping(patience=2),      # 出现梯度爆炸或消失时停止训练
    ]
)


# 【模型评估】
# ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■
score = model.evaluate(X_test, Y_test, verbose=0)
print("score[loss, accuracy] =", score)


# 【预测】
# ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■
result = model.predict_classes(X_test)
print(y_test.reshape((-1,))[:10])
print(result[:10])


# 【保存模型】
# ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■
model.save("./dump/model.h5")

"""【模型结构、模型参数、优化器参数】
    # 保存
    model.save("./dump/model.h5")

    # 加载
    model = load_model("./dump/model.h5")
"""

"""【模型结构】
    # 保存
    with open("./dump/struct.json", "w") as f:
        f.write(model.to_json())

    # 加载 (返回未编译的模型)
    with open("./dump/struct.json", "r") as f:
        json_str = ''.join(f.readlines())
        model = model_from_json(json_str)
"""

"""【模型参数】
    # 保存
    model.save_weights("./dump/model_weights.h5")

    # 加载（不能继续训练模型）
    model_weights = model.load_weights("./dump/model_weights.h5")
"""


# 【总结模型】
# ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■
model.summary()  # 打印模型结构
