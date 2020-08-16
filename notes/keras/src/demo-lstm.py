import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, Activation, LSTM
from keras.losses import mean_squared_error as keras_mse
from keras.optimizers import adam as keras_adam

# 模型存储位置
import os
DUMP_PATH = './dump'
if not os.path.exists(DUMP_PATH):
    os.makedirs(DUMP_PATH)
MODEL_FILE = f"{DUMP_PATH}/model_lstm.h5"


def load_data():
    """
    1949-01~1960-12共12年的每月国际航空公司的乘客人数
    :return: [(year, month), ...], [(number)]
    """
    seq_time = [(_i.year, _i.month) for _i in
                pd.date_range(start='1949-01-01', end='1960-12-31', freq='M')]
    seq_number = [
        112., 118., 132., 129., 121., 135., 148., 148., 136., 119., 104., 118.,
        115., 126., 141., 135., 125., 149., 170., 170., 158., 133., 114., 140.,
        145., 150., 178., 163., 172., 178., 199., 199., 184., 162., 146., 166.,
        171., 180., 193., 181., 183., 218., 230., 242., 209., 191., 172., 194.,
        196., 196., 236., 235., 229., 243., 264., 272., 237., 211., 180., 201.,
        204., 188., 235., 227., 234., 264., 302., 293., 259., 229., 203., 229.,
        242., 233., 267., 269., 270., 315., 364., 347., 312., 274., 237., 278.,
        284., 277., 317., 313., 318., 374., 413., 405., 355., 306., 271., 306.,
        315., 301., 356., 348., 355., 422., 465., 467., 404., 347., 305., 336.,
        340., 318., 362., 348., 363., 435., 491., 505., 404., 359., 310., 337.,
        360., 342., 406., 396., 420., 472., 548., 559., 463., 407., 362., 405.,
        417., 391., 419., 461., 472., 535., 622., 606., 508., 461., 390., 432.]
    return (
        np.array(seq_time, dtype=np.int32),
        np.array(seq_number, dtype=np.float32).reshape((-1, 1))
    )


"""
    读取数据
"""
x_raw, y_raw = load_data()


"""
    数据预处理
"""
# 标准化
x_std = StandardScaler()
x_data = x_std.fit_transform(x_raw)

y_std = StandardScaler()
y_data = y_std.fit_transform(y_raw)


def build_timesteps(_x, _y, look_back=1, step=1):
    sequence = [(_i-look_back, _i) for _i in range(look_back, _x.shape[0] + 1, step)]
    _x_buf, _y_buf = [], []
    for seq in sequence:
        _x_buf.append(_x[seq[0]:seq[1]])
        _y_buf.append(_y[seq[1]-1])
    return np.array(_x_buf), np.array(_y_buf)


# 构建step
BATCH_SIZE = 6
TIME_STEP = 2
x_data, y_data = build_timesteps(x_data, y_data, TIME_STEP)
INPUT_SHAPE = [TIME_STEP, x_data.shape[-1]]
# print(x_data.shape)
# print(y_data.shape)


# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.25, shuffle=False)
# plt.plot(Y_train.reshape(-1))
# plt.plot(Y_test.reshape(-1))
# plt.show()
# print(X_train.shape)
# print(X_test.shape)


# ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■
# ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■


if os.path.exists(MODEL_FILE):
    """
        模型加载
    """
    model = load_model(MODEL_FILE)
else:
    """
        构建模型与编译（首次）
        LSTM-Cell的输入格式为 [samples, time_steps, features]
            return_sequences: False-返回最后一步的状态; True-返回每一步的状态
            return_state: True-返回最后一步的状态;
    """
    # LSTM
    model = Sequential()
    model.add(LSTM(units=4, input_shape=INPUT_SHAPE, return_sequences=True))
    model.add(LSTM(units=2, return_sequences=False))
    model.add(Dense(1))
    model.compile(loss=keras_mse, optimizer=keras_adam(lr=0.01), metrics=['mae'])

    """
        训练与保存（首次）
    """
    model.fit(
        X_train, Y_train,
        epochs=100, batch_size=BATCH_SIZE,
        validation_data=(X_test, Y_test)
    )
    model.save(MODEL_FILE)

# ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■
# ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■

"""
    模型评估
"""
score = model.evaluate(X_test, Y_test, batch_size=32, verbose=0)
print("score[loss, mae] =", score)


"""
    预测结果
"""
fig = plt.figure(figsize=[8.4, 4.8])

y_train_pred = model.predict(X_train, BATCH_SIZE)
ax00 = fig.add_subplot(121)
ax00.set_title("Train")
ax00.plot(Y_train, label='real')
ax00.plot(y_train_pred, label='pred')
ax00.legend()
ax00.grid()

y_test_pred = model.predict(X_test, BATCH_SIZE)
ax01 = fig.add_subplot(122)
ax01.set_title("Test")
ax01.plot(Y_test, label='real')
ax01.plot(y_test_pred, label='pred')
ax01.legend()
ax01.grid()

plt.show(figure=fig)
