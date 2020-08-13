<link rel="stylesheet" href="https://zhmhbest.gitee.io/hellomathematics/style/index.css">
<script src="https://zhmhbest.gitee.io/hellomathematics/style/index.js"></script>

# [Keras](../index.html)

[TOC]

## Reference

- [Keras API](https://keras.io/api/)
- [中文文档](https://keras.io/zh/)

## Hello

>[`demo-Hello.py`](./src/demo-hello.py)

```flow
st=>start: 开始

data=>inputoutput: 数据载入
preprocessing=>operation: 数据预处理

model=>operation: 定义模型
compile=>operation: 编译模型
training=>operation: 训练模型
assessment=>operation: 评估模型

predict=>operation: 预测
save=>operation: 保存模型
ed=>end: 结束

st->data->preprocessing->model->compile->training->assessment->predict->save->ed
```

## 定义

>- [Layers](https://keras.io/api/layers/)

通过`add`方法可以向模型添加各式网络。

需要注意的是，第一层应指定输入维度（`input_dim`）使其与数据的特征数量一致，最后一层的单元数量（`units`）应与输出维度相同。

```py
from keras.models import Sequential
from keras.layers import (
    Dense, Activation, Dropout,
    Conv2D, AvgPool2D, MaxPool2D, Flatten,
    SimpleRNN, LSTM, GRU,
)

model = Sequential()
model.add(Dense(units=?, input_dim=?))
model.add(Activation('relu'))
# ...
```

## 编译

>- [Losses](https://keras.io/zh/losses/)
>- [Optimizers](https://keras.io/zh/optimizers/)
>- [Metrics](https://keras.io/zh/metrics/)

对定义后的模型进行编译，可选择合适的损失函数和优化器。

```py
# 损失函数
from keras.losses import (
    mean_squared_error as keras_mse,
    categorical_crossentropy as keras_ce
)
# 优化器
from keras.optimizers import (
    adam as keras_adam,
    sgd as keras_sgd
)
# 评估指标
from keras.metrics import (
    categorical_accuracy, mae, mse
)  # = 'acc', 'mae', 'mse'

model.compile(
    loss=keras_mse,
    optimizer=keras_adam(lr=0.01),
    metrics=[
        # categorical_accuracy, mse, mae
    ]
)
```

## 训练

### 自动训练

>- [Callback](https://keras.io/zh/callbacks/)

```py
from keras.callbacks import Callback as keras_Callback
class FitCall(keras_Callback):
    def on_epoch_begin(self, epoch, logs=None):
        # 记录是第几轮训练
        print('epoch =', epoch)

model.fit(
    x_train, y_train,  # 训练数据
    # validation_data=(x_test, y_test),  # 评估用
    batch_size=32, epochs=500,
    verbose=0,  # 0,1,2 = 无、进度条、第几轮
    callbacks=[
        FitCall()
    ]
)
```

### 自助训练

```py
for step in range(2000):
    loss_val = model.train_on_batch(x_batch, y_batch)
    if 0 == step % 200:
        print('loss:', loss_val)
```

## 评估

模型评估将代入测试数据，计算损失值和在编译时指定的其它评估指标。

```py
score = model.evaluate(x_test, y_test, batch_size=32, verbose=0)
print("score[loss, ...] =", score)
```

## 预测

```py
# y_pred = model.predict_classes(x_test)
y_pred = model.predict(x_test)
# 比较 y_test 和 y_pred
print(y_test[:10])
print(y_pred[:10])
```

## 保存

```py
from keras.models import Sequential
from keras.models import (load_model, model_from_json)

"""
    【模型结构】 可重新实例化模型
    【模型权重】 可应用模型
    【优化器状态】 可继续训练
"""
# 保存（模型定义、编译、训练后）
model.save("./dump/model.h5")

# 加载（模型定义前，即不需要再定义和编译，若需要可继续训练）
model = load_model("./dump/model.h5")


"""
    【模型结构】
"""
# 在命令窗口中打印模型结构
model.summary()

# 保存（模型定义后）
with open("./dump/struct.json", "w") as fp:
    fp.write(model.to_json())

# 加载 (返回未编译的模型)
with open("./dump/struct.json", "r") as fp:
    model = model_from_json(fp.read())


"""
    【模型权重】
"""
# 保存（模型定义、编译、训练后）
model.save_weights("./dump/model_weights.h5")

# 加载（不能继续训练模型）
model = model.load_weights("./dump/model_weights.h5")
```
