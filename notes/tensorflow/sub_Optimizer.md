## Optimizer

>[参考资料](https://ruder.io/optimizing-gradient-descent/index.html)

![](./images/optimizer1.gif)
![](./images/optimizer2.gif)

<!-- https://blog.csdn.net/u010089444/article/details/76725843 -->

### StochasticGradientDescent（SGD）

每读入一个数据，便立刻计算Loss的梯度来更新参数。

- **优点**：有几率跳出局部最优。
- **缺点**：可能被困在鞍点（此点处代价震荡）。

```py
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)
```

### Momentum

模拟物体运动时的惯性，即更新的时候在一定程度上保留之前更新的方向。

```py
train_op = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(loss_function)
```

### Nesterov Accelerated Gradient（NAG）

我们希望有一个更聪明的球，它知道在山坡再次变缓之前会减速。

```py
# None
```

### Adagrad

可以使学习速率适应参数，对低频特征做较大更新，对高频的做较小更新。

- **优点**：无需手动调整学习速度。在稀疏数据上的表现优异。
- **缺点**：分母会不断积累，导致学习率急速下降并快速趋近于零。

```py
train_op = tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=0.01).minimize(loss_function)
```

### Adadelta

是对Adagrad的改进，Adadelta不会累计过去所有的平方梯度，历史梯度的积累将会被限制在某个固定大小，从而避免学习率的急速下降。该算法甚至不需要设置默认学习率。

```py
train_op = tf.train.AdadeltaOptimizer(rho=0.95).minimize(loss_function)
```

### RMSProp

RMSprop和Adadelta都是为了解决Adagrad学习率急剧下降问题的。

```py
train_op = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.9).minimize(loss_function)
```

### Adam

相当于RMSprop + Momentum，训练过程就像是一个带有摩擦的沉重的球。

```py
train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(loss_function)
```
