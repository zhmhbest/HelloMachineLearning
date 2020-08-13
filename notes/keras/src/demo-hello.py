import numpy as np
from keras.models import Sequential, load_model
from keras.layers import (Dense, Activation)
from keras.metrics import categorical_accuracy, mae, mse
from keras.losses import mean_squared_error
from keras.optimizers import sgd
from keras.callbacks import Callback as keras_Callback
from matplotlib import pyplot
import os
DUMP_PATH = './dump'
if not os.path.exists(DUMP_PATH):
    os.makedirs(DUMP_PATH)
MODEL_FILE = f"{DUMP_PATH}/model_hello.h5"


"""
    模拟数据
"""
x_data = np.random.rand(128).reshape(-1, 1)
y_data = np.random.normal(
    (lambda x: np.sqrt(-x ** 2 + x * 0.5 + 0.8))(x_data),
    0.008
)


if os.path.exists(MODEL_FILE):
    """
        模型加载
    """
    model = load_model(MODEL_FILE)
else:
    """
        模型定义与编译（首次）
    """
    model = Sequential()
    model.add(Dense(8, input_dim=x_data.shape[1]))
    model.add(Activation('tanh'))
    model.add(Dense(4))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('tanh'))
    model.compile(
        loss=mean_squared_error,
        optimizer=sgd(lr=0.1),
        metrics=[
            mse, mae
        ]
    )
    """
        模型训练（首次）
    """
    class FitCall(keras_Callback):
        def on_epoch_begin(self, epoch, logs=None):
            print('epoch =', epoch)
    model.fit(
        x_data, y_data,  # 训练数据
        # validation_data=(x_test, y_test),  # 评估用
        batch_size=32, epochs=500,
        verbose=0,  # 0,1,2 = 无、进度条、第几轮
        callbacks=[
            FitCall()
        ]
    )
    """
        模型保存（首次）
    """
    model.save(MODEL_FILE)


"""
    模型评估
"""
score = model.evaluate(x_data, y_data, batch_size=32, verbose=0)
print("score[loss, mse, mae] =", score)


"""
    打印模型结构
"""
model.summary()


"""
    模型预测
"""
y_pred = model.predict(x_data)
pyplot.scatter(x_data, y_data, label='data')
pyplot.scatter(x_data, y_pred, label='pred')
pyplot.grid()
pyplot.legend()
pyplot.show()
