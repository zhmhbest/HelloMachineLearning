"""
    待补充
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.losses import mean_squared_error as keras_mse
from keras.optimizers import adagrad as keras_adagrad
from sklearn. preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import math
import pylab

"""
    模型定义与编译
"""
model = Sequential()
model.add(LSTM(10, input_shape=()))
model.add(Dense(1))
model.compile(loss=keras_mse, optimizer=keras_adagrad)
