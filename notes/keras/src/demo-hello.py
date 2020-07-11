import numpy as np
import pandas as pd
from keras.datasets import cifar10
from keras.models import Sequential, load_model, model_from_json
from keras.layers.core import Dense, Dropout, Activation
"""
    Dense       : 用于创建密集神经元
    Dropout     : 用于正则化
    Activation  : 用于创建激活层
"""
from keras.optimizers import adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from zhmh import make_dump
make_dump('./dump')  # 创建缓存目录


# 【数据载入】
# ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■
"""
    cifar10 是一个包含阿拉伯数字图片的数据集
        数据类型: ndarray
        训练数目: 50000
        测试数目: 10000
        图片尺寸: 32 × 32 × 3
        输出结果: len(0,1,2,3,4,5,6,7,8,9) = 10
"""
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
PICTURE_SIZE = 32 * 32 * 3
TRAIN_SIZE = x_train.shape[0]
TEST_SIZE = x_test.shape[0]
OUTPUT_SIZE = 10


# 【数据预处理】
# ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■
def gaussian_distribution(d):
    """
    高斯分布
    """
    return (d - np.mean(d)) / np.std(d)


X_train = gaussian_distribution(x_train.reshape(TRAIN_SIZE, PICTURE_SIZE))
X_test = gaussian_distribution(x_test.reshape(TEST_SIZE, PICTURE_SIZE))
print(X_train.shape, X_test.shape)
# One-Hot 编码
Y_train = np_utils.to_categorical(y_train, OUTPUT_SIZE)
Y_test = np_utils.to_categorical(y_test, OUTPUT_SIZE)
print(y_train[0:3])
print(Y_train[0:3])
print(Y_train.shape, Y_test.shape)


# 【模型定义】
# ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■
model = Sequential()
model.add(Dense(512, input_shape=(PICTURE_SIZE,)))
model.add(Activation('relu'))       # 激活函数
model.add(Dropout(0.4))             # 正则化
model.add(Dense(120))
model.add(Activation('relu'))       # 激活函数
model.add(Dropout(0.2))             # 正则化
model.add(Dense(OUTPUT_SIZE))       # 输出层
model.add(Activation('sigmoid'))    # 激活函数


# 【模型编译】
# ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■
# 定义：损失函数、优化器
model.compile(
    loss='categorical_crossentropy',
    optimizer=adam(0.01),
    metrics=['accuracy']
)


# 【模型训练】
# ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■

model.fit(
    X_train, Y_train,
    batch_size=1000,                    # 每组数量
    nb_epoch=10,                        # 循环次数
    validation_data=(X_test, Y_test),
    callbacks=[
        EarlyStopping(patience=2)       # 出现梯度爆炸或消失时停止训练
    ]
)


# 【模型评估】
# ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■
score = model.evaluate(X_test, Y_test, verbose=0)
print("score[loss, accuracy] =", score)

# 【预测】
# ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■
model.predict_classes(X_test)


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
