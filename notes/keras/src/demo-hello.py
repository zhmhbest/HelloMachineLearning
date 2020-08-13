import numpy as np
from keras.models import Sequential
from keras.models import (load_model, model_from_json)
from keras.layers.core import (Dense, Activation)
from keras.optimizers import sgd
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
        loss='mse',
        optimizer=sgd(lr=0.1)
    )

    """
        模型训练（首次）
    """
    for step in range(2000):
        loss_val = model.train_on_batch(x_data, y_data)
        if 0 == step % 200:
            print('loss:', loss_val)

    """
        模型评估与保存（首次）
    """
    score = model.evaluate(x_data, y_data, verbose=0)
    print("score[loss] =", score)
    model.save(MODEL_FILE)


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
