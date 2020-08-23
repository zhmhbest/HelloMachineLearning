<link rel="stylesheet" href="https://zhmhbest.gitee.io/hellomathematics/style/index.css">
<script src="https://zhmhbest.gitee.io/hellomathematics/style/index.js"></script>

# [Keras](../index.html)

[TOC]

## Reference

- [Keras API](https://keras.io/api/)
- [中文文档](https://keras.io/zh/)

## Hello

```flow
st=>start: 开始

data=>inputoutput: 数据载入
preprocessing=>operation: 数据预处理

model=>operation: 定义模型
compile=>operation: 编译模型
training=>operation: 训练模型
assessment=>operation: 评估模型

predict=>inputoutput: 预测数据
save=>operation: 保存模型
ed=>end: 结束

st->data->preprocessing(right)->model
model->compile(right)->training
training->save(right)->assessment
assessment->predict->ed
```

### 定义模型

>- [Layers](https://keras.io/api/layers/)

通过`add`方法可以向模型添加各式网络。

需要注意的是，第一层应指定输入维度（`input_dim`）使其与数据的特征数量一致，最后一层的单元数量（`units`）应与输出维度相同。

#### 顺序模型

```py
from keras.models import Sequential, load_model
from keras.layers import (
    Dense, Activation, Dropout,
    Conv2D, MaxPooling2D, AveragePooling2D, Flatten,
    SimpleRNN, LSTM, GRU
)

model = Sequential()
model.add(Dense(units=?, input_dim=?))
model.add(Activation('relu'))
# ...
```

#### 自定义模型

```py
from keras.models import Sequential
from keras.models import Input, Model
from keras.layers import Dense, Activation

# 不包括batch_size
INPUT_SHAPE = (2,)

# 顺序模型
model1 = Sequential()
model1.add(Dense(4, input_shape=INPUT_SHAPE))
model1.add(Activation('relu'))
model1.add(Dense(1))

# 自定义模型
inputs = Input(shape=INPUT_SHAPE)
x = inputs
x = Dense(4)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
y = x
model2 = Model(inputs=inputs, outputs=y)
```

#### 混合模型

```py
from keras.backend import concatenate
from keras.models import Sequential
from keras.layers import Dense, Activation

# 不包括batch_size
INPUT_SHAPE_A = (32,)
INPUT_SHAPE_B = (128,)

partA = Sequential()
partA.add(Dense(8, input_shape=INPUT_SHAPE_A))
partA.add(Activation('relu'))
partA.add(Dense(4))
print(partA.input.shape, partA.output.shape)

partB = Sequential()
partB.add(Dense(32, input_shape=INPUT_SHAPE_B))
partB.add(Activation('relu'))
partB.add(Dense(16))
print(partB.input.shape, partB.output.shape)

combined = concatenate([partA.output, partB.output])
next_shape = (lambda x: (x.pop(0), tuple(x)))(combined.shape.as_list())[1]
print(combined.shape, next_shape)

model = Sequential()
model.add(Dense(2, input_shape=next_shape))
model.add(Activation('relu'))
model.add(Dense(1))
print(model.input.shape, model.output.shape)
```

### 编译模型

>- [Losses](https://keras.io/zh/losses/)
>- [Optimizers](https://keras.io/zh/optimizers/)
>- [Metrics](https://keras.io/zh/metrics/)

对定义后的模型进行编译，可选择合适的损失函数和优化器。

```py
# 损失函数
from keras.losses import (
    # 回归问题
    mean_absolute_error as keras_loss_mae,
    mean_squared_error as keras_loss_mse,
    # 分类问题
    binary_crossentropy as keras_loss_bce,              # targets为one-hot、输出层使用sigmoid激活
    categorical_crossentropy as keras_loss_cce,         # targets为one-hot、输出层使用softmax激活
    sparse_categorical_crossentropy as keras_loss_sce   # targets为number 、输出层使用softmax激活
)
# 优化器
from keras.optimizers import (
    adam as keras_optimizer_adam,
    sgd as keras_optimizer_sgd
)
# 评估指标（Evaluator）
from keras.metrics import binary_accuracy, categorical_accuracy

model.compile(
    loss=keras_loss_mse,
    optimizer=keras_optimizer_adam(lr=0.01),
    metrics=[
        # binary_accuracy, categorical_accuracy
    ]
)
```

### 训练模型

#### 自动训练

>- [Callback](https://keras.io/zh/callbacks/)

```py
from keras.callbacks import Callback as keras_Callback
class MyCallback(keras_Callback):
    def on_epoch_begin(self, epoch, logs=None):
        # 记录是第几轮训练
        print('epoch =', epoch)

model.fit(
    x_train, y_train,  # 训练数据
    # validation_data=(x_test, y_test),  # 评估用
    batch_size=32, epochs=500,
    verbose=0,  # 0,1,2 = 无、进度条、第几轮
    callbacks=[
        MyCallback()
    ]
)
```

#### 自助训练

```py
for step in range(2000):
    loss_val = model.train_on_batch(x_batch, y_batch)
    if 0 == step % 200:
        print('loss:', loss_val)
```

### 保存模型

```py
from keras.models import Sequential, load_model, model_from_json

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

### 评估模型

模型评估将代入测试数据，计算损失值和在编译时指定的其它评估指标。

```py
score = model.evaluate(x_test, y_test, batch_size=32, verbose=0)
print("score[loss, ...] =", score)
```

### 预测数据

```py
# y_pred = model.predict_classes(x_test)    # 自动将one-hot转换为number
y_pred = model.predict(x_test)              # 原样数据

# 比较 y_test 和 y_pred 的前10个数据
print(y_test[:10])
print(y_pred[:10])
```

## Backend

>- [Backend](https://keras.io/zh/backend/)

```py
import keras
from keras import backend as K
```

### Placeholder

```py
# (samples, features)
inputs = K.placeholder(shape=(None, features))

# (samples, time_step, features)
inputs = K.placeholder(shape=(None, time_step, features))
```

### Tensor

```py
# Variable
variable_shape = (3, 4, 5)
var0 = K.zeros(shape=variable_shape)
var1 = K.ones(shape=variable_shape)
# K.variable(value, dtype=None, name=None, constraint=None)
var2 = K.variable(value=np.random.random(size=variable_shape))
var3 = K.random_uniform_variable(shape=variable_shape, low=0, high=1)   # 均匀分布
var4 = K.random_normal_variable(shape=variable_shape, mean=0, scale=1)  # 高斯分布

# Constant
# K.constant(value, dtype=None, shape=None, name=None)
val0 = K.constant(value=0)
```

### Layer

>[Own Layer](https://keras.io/zh/layers/writing-your-own-keras-layers/)

```py
from keras import backend as K
from keras.layers.core import Lambda as keras_Lambda
# eg: model.add(Lambda(lambda x: x ** 2))
from keras.engine.topology import Layer as keras_Layer


class SimpleDense(keras_Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.kernel = None
        self.bias = None

    def build(self, input_shape):
        # 确定可训练参数
        from keras import initializers, regularizers, constraints
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_dim, self.units),
            initializer=initializers.get('glorot_uniform')
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer=initializers.get('zeros')
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        # 在此处编写逻辑
        return K.bias_add(
            K.dot(inputs, self.kernel),
            self.bias
        )

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)


if __name__ == '__main__':
    x = K.placeholder(shape=(None, 2))
    y = SimpleDense(units=3)(x)
    print(y.shape)
```

## Demonstrate

[`demo-Hello.py`](./src/demo-hello.py)

[`demo-CNN.py`](./src/demo-cnn.py)

[`demo-LSTM.py`](./src/demo-lstm.py)
