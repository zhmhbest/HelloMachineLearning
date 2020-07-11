<link rel="stylesheet" href="https://zhmhbest.gitee.io/hellomathematics/style/index.css">
<script src="https://zhmhbest.gitee.io/hellomathematics/style/index.js"></script>

# [神经网络](../index.html)

[TOC]

## Hello

>[`demo-HelloNeuralNetworks.py`](./src/demo-HelloNeuralNetworks.py)

```flow
st=>start: 开始
config=>inputoutput: 参数设置
networks=>operation: 定义神经网络
loss=>operation: 定义损失函数
optimizer=>operation: 定义优化器
training=>operation: 训练
ed=>end: 结束

st->config->networks->loss->optimizer->training->ed
```

## 学习率

>[`demo-LearningRate.py`](./src/demo-LearningRate.py)

学习率既不能过大，也不能过小。 过小：训练速度慢；过大：可能导致模型震荡。

**指数衰减法**则可很好的解决上述问题

```py
"""
learning_rate = initial_learning_rate * decay_rate ^ (global_step / decay_steps)
    learning_rate        : 每轮实际使用的学习率
    initial_learning_rate: 初始学习率
    global_step          : tf.Variable(0, trainable=False)
    decay_steps          : 衰减速度
    decay_rate           : 衰减系数
    staircase            : 是否以离散间隔衰减学习速率

global_step为固定写法，用以记录训练次数
staircase为真时(global_step / decay_steps)的值会被转化为整数
"""
decayed_learning_rate = tf.train.exponential_decay(
    initial_learning_rate,
    global_step,
    decay_steps,
    decay_rate,
    staircase=False
)
```

## 拟合

>[`demo-Fitting.py`](./src/demo-Fitting.py)

使用正则化避免过拟合。

## 滑动平均模型

>[`demo-Mnist.py`](./src/demo-Mnist.py)

滑动平均模型用来估计变量的局部均值，使得变量的更新与一段时间内的历史取值有关。
在采用随机梯度下降算法训练神经网络时，使用滑动平均模型可以在一定程度提高最终模型在测试数据上的表现。

该模型会维护一个影子变量（shadow variable），这个影子变量的初始值就是相应变量的初始值，每次运行时，会对影子变量的值进行更新。

```py
"""
shadow_variable = decay * shadow_variable + (1 - decay) * variable
    decay:                  衰减率，决定更新速度（一般为0.999或0.9999）
    shadow_variable:        影子变量
    variable:               待更新的变量
"""

ema = tf.train.ExponentialMovingAverage(decay=衰减率, num_updates=None)
    # num_updates(optional):  动态设置decay大小
    # decay = min(decay, (1+num_updates)/(10+num_updates))
x = activation(tf.matmul(x, weight) + biases)
x = activation(tf.matmul(x, ema.average(weight)) + ema.average(biases))
```

## 模型复用

>[`demo-ModelReuse.py`](./src/demo-ModelReuse.py)
