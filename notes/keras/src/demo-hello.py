import numpy as np
from keras.models import Sequential
from zhmh.dataset import generate_random_data
from keras.layers.core import (Dense, Dropout, Activation)
from keras.losses import mean_squared_error
from keras.optimizers import adam
from sklearn.model_selection import train_test_split
from matplotlib import pyplot


"""
    模拟数据
"""
data_x, data_y = generate_random_data(512, 2, 1)
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)

"""
    模型定义
"""
model = Sequential()
model.add(Dense(4, input_dim=x_train.shape[1]))
model.add(Activation('elu'))
model.add(Dense(2, input_dim=x_train.shape[1]))
model.add(Activation('elu'))
model.add(Dense(1))


"""
    编译模型
"""
model.compile(
    loss=mean_squared_error,
    optimizer=adam(lr=0.01)
)


"""
    模型训练
"""
model.fit(
    x_train, y_train,
    batch_size=32,      # 每组数量
    epochs=50,         # 循环次数
    validation_data=(x_test, y_test),
)


"""
    模型评估
"""
score = model.evaluate(x_test, y_test, verbose=0)
print("score[loss] =", score)


"""
    模型预测
"""
test_pred = model.predict_classes(x_test)
pyplot.plot(test_pred)
pyplot.plot(y_test)
pyplot.show()

