import os
import numpy as np
# 数据集
from keras.datasets import cifar10
# One-Hot
from keras.utils.np_utils import to_categorical
from keras.optimizers import adam
from keras.callbacks import EarlyStopping
from keras.activations import relu, softmax
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.models import Sequential, load_model
from keras.constraints import MaxNorm
from keras.layers.core import (Dense, Dropout, Activation)
"""
    Dense           : 用于创建密集神经元
    Dropout         : 用于正则化
    Activation      : 用于创建激活层
"""

from keras.layers import (
    Conv2D,
    MaxPooling2D, AveragePooling2D,
    Flatten
)
"""
    【卷积层】
    Conv2D
        filters             : {int} 过滤器输出深度
        kernel_size         : {int | (int, int)} 过滤器窗口尺寸
        strides             : {int | (int, int)} 过滤器过滤步长
        padding             : {'valid' | 'same'} 过滤器模式
            valid = 无填充
            same  = 填充输入，使输出与原始输入的长度相同
        data_format         : {'channels_last' | 'channels_first'} 数据格式
            channels_last
                input_shape=(samples, rows, cols, channels)
                output_shape=(samples, new_rows, new_cols, filters)
            channels_first
                input_shape=(samples, channels, rows, cols)
                output_shape=(samples, filters, new_rows, new_cols)
        dilation_rate       : {int | (int, int)} 过滤器扩展
        activation          : {str | keras.activations.?} 激活函数
        use_bias            : {bool} 是否使用偏置
        kernel_constraint   : {keras.constraints.?} 约束主权重矩阵
            ?=MaxNorm       最大范数权值约束
            ?=NonNeg        权重非负的约束
            ?=UnitNorm      映射到每个隐藏单元的权值的约束，使其具有单位范数
            ?=MinMaxNorm    最小/最大范数权值约束
        bias_constraint     : {keras.constraints.?} 约束偏置

    【池化层】
    MaxPooling2D/AveragePooling2D
        pool_size       : {int | (int, int)} 池化窗口尺寸
        strides         : {int | (int, int)} 池化步长
        padding         : {'valid' | 'same'} 池化模式
        data_format     : {'channels_last' | 'channels_first'} 数据格式
            channels_last
                input_shape=(batch_size, rows, cols, channels)
                output_shape=(batch_size, pooled_rows, pooled_cols, channels)
            channels_first
                input_shape=(batch_size, channels, rows, cols)
                output_shape=(batch_size, channels, pooled_rows, pooled_cols)
    
    【扁平层】
    Flatten         : 特征一维化
        input_shape=(None, n1, n2, ...)
        output_shape=(None, n1 * n2 * ...)
"""

DUMP_PATH = './dump'
if not os.path.exists(DUMP_PATH):
    os.makedirs(DUMP_PATH)
MODEL_FILE = f"{DUMP_PATH}/model_cnn.h5"
CONTINUE_TRAIN = True


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


if os.path.exists(MODEL_FILE):
    # 【模型加载】
    # ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■
    model = load_model(MODEL_FILE)
    print(f'Load "{MODEL_FILE}"')
else:
    # 【模型定义（首次）】
    # ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■
    print('Define model')
    # 准确率最高可达 65%
    model = Sequential()
    model.add(Conv2D(8, (2, 2), input_shape=INPUT_SHAPE, padding='same', activation=relu, kernel_constraint=MaxNorm(3)))
    model.add(Conv2D(16, (3, 3), activation=relu, padding='same', kernel_constraint=MaxNorm(3)))
    model.add(Conv2D(32, (4, 4), activation=relu, padding='same', kernel_constraint=MaxNorm(3)))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (2, 2), activation=relu, padding='same', kernel_constraint=MaxNorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation(relu))
    model.add(Dropout(0.2))
    model.add(Dense(16))
    model.add(Activation(relu))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation(softmax))

    # 【模型编译（首次）】
    # ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■
    # 定义：损失函数、优化器
    model.compile(
        loss=categorical_crossentropy,
        optimizer=adam(lr=0.01),
        metrics=[categorical_accuracy]
    )


if CONTINUE_TRAIN:
    # 【模型训练】
    # ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■
    model.fit(
        X_train, Y_train,
        batch_size=1000,                    # 每组数量
        epochs=10,                          # 循环次数
        validation_data=(X_test, Y_test),
        verbose=2,
        callbacks=[
            EarlyStopping(patience=2, verbose=2),   # 出现梯度爆炸或消失时停止训练
        ]
    )

    # 【保存模型】
    # ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■
    model.save(MODEL_FILE)
    print(f'Saved to "{MODEL_FILE}"')


# 【模型评估】
# ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■
score = model.evaluate(X_test, Y_test, verbose=0)
print("score[loss, accuracy] =", score)


# 【预测】
# ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■
# y_pred = model.predict(X_test)
y_pred = model.predict_classes(X_test)
print(y_test[:10].reshape((-1,)))
print(y_pred[:10])
