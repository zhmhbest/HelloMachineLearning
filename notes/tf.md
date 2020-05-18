<link rel="stylesheet" href="https://zhmhbest.gitee.io/hellomathematics/style/index.css">
<script src="https://zhmhbest.gitee.io/hellomathematics/style/index.js"></script>
<link rel="stylesheet" href="..\style\codeblock.css">

# [Tensorflow入门](./index.html)

[TOC]

## Hello

```py
import tensorflow as tf

# Tensor
a = tf.constant(1)
b = tf.constant(2)
result = tf.add(a, b)
print(result)


# Session
with tf.Session() as sess:
    print(sess.run(result))
```

## Tensor

- $0$阶张量：标量（*Scalar*）也就是$1$个实数
- $1$阶张量：向量（*Vector*）也就是$1$维数组
- $2$阶张量：矩阵（*Matrix*）也就是$2$维数组
- $n$阶张量：$n$维数组

### 张量的属性

**定义一个常量张量**

```py
tensor_constant_demo = tf.constant([
    [1.1, 2.2, 3.3],
    [4.4, 5.5, 6.6]
], name='tensor_constant_demo', dtype=tf.float32)
```

**Operation（OP，运算）**：节点在图中被称为OP，OP即某种抽象计算。

```py
print(tensor_constant_demo.op)
print("name" , tensor_constant_demo.name)
print("type" , tensor_constant_demo.dtype)
print("shape", tensor_constant_demo.shape)
print("graph", tensor_constant_demo.graph)
```

### 调整张量

```py
# 类型转换
result_cast = tf.cast(tensor_constant_demo, tf.int32, name='result_cast')
print(tensor_constant_demo.dtype, '=>', result_cast.dtype)

# 结构调整
result_reshape = tf.reshape(tensor_constant_demo, [3, 2], name='result_reshape')
print(tensor_constant_demo.shape, '=>', result_reshape.shape)
```

### 序列张量

```py
# 从1开始步长为3不超过10的序列
tensor_range = tf.range(1, 10, 3, dtype=None, name=None)
'''
    [1 4 7]
'''

# 10~100等分为5份
tensor_space = tf.linspace(10.0, 100.0, 5, name=None)
'''
    [10.0 32.5 55.0 77.5 100.0]
'''
```

### 填充张量

```py
# 产生以给定值填充的张量
tensor_fill = tf.fill([2, 3], 99, name=None)
'''
[[99 99 99]
 [99 99 99]]
'''

# 产生以0填充的张量
tensor_zeros = tf.zeros([2, 3], tf.float32, name=None)
'''
[[0. 0. 0.]
 [0. 0. 0.]]
'''

# 产生以1填充的张量
tensor_ones = tf.ones([2, 3], tf.float32, name=None)
'''
[[1. 1. 1.]
 [1. 1. 1.]]
'''

# 产生对角线为[1, 2, 3, 4]其余为0的二维张量
tensor_diag = tf.diag([1, 2, 3, 4], name=None)
'''
[[1 0 0 0]
 [0 2 0 0]
 [0 0 3 0]
 [0 0 0 4]]
'''
```

### 随机数张量

```py
# 正态分布随机数(shape, mean:平均数, stddev:标准差)
tensor_random1 = tf.random_normal([2, 3], 10, 0.6, name=None)
'''
[[9.929678 9.88656  9.663629]
 [9.634826 9.379279 9.33766 ]]
'''

# 正态分布随机数，偏离2个标准差的随机值会被重新生成
tensor_random2 = tf.truncated_normal([2, 3], 10, 0.6, name=None)
'''
[[10.298693 10.121988 10.665423]
 [10.015184 10.673774 10.18005 ]]
'''

# 均匀分布随机数(shape, min, max)
tensor_random3 = tf.random_uniform([2, 3], 1, 10, name=None)
'''
[[8.568491  6.56831   2.8412023]
 [3.7498274 4.389385  8.6796055]]
'''

# Γ(Gamma)随机数(shape, alpha, beta)
# tf.random_gamma(...)
```

## Session

```py
a = tf.constant(1, name="a")
b = tf.constant(2, name="b")
result = tf.add(a, b)
print(result)
# Tensor("Add:0", shape=(), dtype=int32)
```

### 一般会话

```py
with tf.Session() as sess:
    # 方式1
    print(sess.run(result))
    # 方式2
    print(result.eval(session=sess))
```

### 默认会话

```py
with tf.Session().as_default():
    print(result.eval())
```

### 交互会话

该方法一般用在命令窗口中。

```py
sess = tf.InteractiveSession()
print(result.eval())
sess.close()
```

### 配置会话

```py
"""
    log_device_placement: 打印设备信息
    allow_soft_placement: GPU异常时，可以调整到CPU执行
"""
with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    print(sess.run(result))
```

### 运行时数据

```py
X = tf.placeholder(tf.float32, shape=(None, 2))  # 样本组数不固定，每组两个
Y = X * 3
with tf.Session() as sess:

    # → print(sess.run(Y))
    # ↑ InvalidArgumentError: 此处X还没有赋值

    print(sess.run(Y, feed_dict={
        X: [
            [1, 2],
            [2, 3]
        ]
    }))

    print(sess.run(Y, feed_dict={
        X: [
            [1, 2],
            [2, 3],
            [5, 3]
        ]
    }))
```

## Variable

### 定义变量

```py
# 方式1
var1 = tf.Variable(99, name='var1')
var2 = tf.Variable(tf.random_normal([2, 3], mean=10, stddev=2), name='var2')

# 方式2
var3 = tf.get_variable('var3', initializer=tf.zeros_initializer(), shape=[2])
var4 = tf.get_variable('var4', initializer=tf.ones_initializer(), shape=[3])
```

当设置`reuse=True`时，`get_variable`可以防止重复定义。
此外，当使用`get_variable`方法时，变量只能以

- `tf.constant_initializer()`
- `tf.random_normal_initializer()`
- `tf.truncated_normal_initializer()`
- `tf.random_uniform_initializer()`
- `tf.uniform_unit_scaling_initializer()`
- `tf.zeros_initializer()`
- `tf.ones_initializer()`

等方法初始化。

### 使用变量

在使用变量前，必须初始化变量，我们可以采取`sess.run(var1.initializer)`方法只初始化在本次会话中要用到的变量，也可以使用`tf.global_variables_initializer().run()`或`sess.run(tf.global_variables_initializer())`方法直接初始化全部变量。

```py
with tf.Session() as sess:
    # 使用前，必须初始化变量
    sess.run(var1.initializer)
    sess.run(var3.initializer)

    # 获得变量的值
    print(sess.run(var1))
    # print(sess.run(var2))  # var1 未被初始化，使用会报错。

    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("var3")))
        # print(sess.run(tf.get_variable("var4")))  # var4 未被初始化，使用会报错。
```

```py
with tf.Session() as sess:
    # 初始化全部变量
    tf.global_variables_initializer().run()
    # sess.run(tf.global_variables_initializer())

    # 获得变量的值
    print(sess.run(var1))
    print(sess.run(var2))
    print()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("var3")))
        print(sess.run(tf.get_variable("var4")))
```

### 变量空间

```py
"""
创建变量
"""
# 创建变量 space1.var1 = 0
with tf.variable_scope("space1"):
    tf.get_variable('var1', initializer=tf.zeros_initializer(), shape=[1])

# 创建变量 space2.var1 = 1
with tf.variable_scope("space2"):
    tf.get_variable('var1', initializer=tf.ones_initializer(), shape=[1])

"""
使用变量
"""
with tf.Session() as sess:
    # 初始化全部变量
    tf.global_variables_initializer().run()

    with tf.variable_scope("space1", reuse=True):
        print(sess.run(tf.get_variable("var1")))

    with tf.variable_scope("space2", reuse=True):
        print(sess.run(tf.get_variable("var1")))
```

## Graph

### 默认图

当我们没有指定图时，Tensorflow会为我们提供一张默认的图，此时，我们的张量、会话、变量都是运行在默认的图上。

```py
g0 = tf.get_default_graph()  # 获取默认的图
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result = a + b

print(a.graph is g0)
print(b.graph is g0)
print(result.graph is g0)
```

### 自定义图

```py
g0 = tf.get_default_graph() 
g1 = tf.Graph()
print("g0 is default?", tf.get_default_graph() is g0)               # True
print("g1 is default?", tf.get_default_graph() is g1)               # False

with g1.as_default():
    print("\t", "g1 is default?", tf.get_default_graph() is g1)     # True
    # g1中定义变量，并赋值
    var1 = tf.Variable(1, name='var1')
    print("\t", "var1 is in g1?", var1.graph is g1)                 # True
# end with(graph)

print("g0 is default?", tf.get_default_graph() is g0)               # True
print("g1 is default?", tf.get_default_graph() is g1)               # False
```

### 自定义图并运行会话

```py
g0 = tf.get_default_graph() 
g2 = tf.Graph()
with g2.as_default():
    print("\t", "g2 is default?", tf.get_default_graph() is g2)           # True
    var1 = tf.Variable(2, name='var1')
    """
        方式一
    """
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print("\t", "g2 var1 = ", sess.run(var1))
    # end with(sess)
# end with(graph)

print("g0 is default?", tf.get_default_graph() is g0)               # True
print("g2 is default?", tf.get_default_graph() is g2)               # False

"""
    方式二
"""
with tf.Session(graph=g2) as sess:
    print("\t", "g2 is default?", tf.get_default_graph() is g2)           # True
    tf.global_variables_initializer().run()
    print("\t", "g2 var1 = ", sess.run(var1))
# end with(sess)
```

## TensorBoard

### 封装

```py
import os
import tensorflow as tf


class TensorBoard:
    def __init__(self, summary_dir="./summary"):
        assert os.path.isdir(os.path.dirname(summary_dir)) is True  # 上级目录存在
        if os.path.exists(summary_dir):
            assert os.path.isdir(summary_dir) is True  # 指定目录不是文件
        else:
            os.makedirs(summary_dir)
        # end if
        self.summary_dir = os.path.abspath(summary_dir)

    def remake(self):
        os.system('RMDIR /S /Q "' + self.summary_dir + '"')
        os.system('MKDIR "' + self.summary_dir + '"')

    def save(self, g):
        tf.summary.FileWriter(self.summary_dir, graph=g)

    def board(self):
        print("TensorBoard may view at:")
        print(" * http://%s:6006/" % os.environ['ComputerName'])
        print(" * http://localhost:6006/")
        print(" * http://127.0.0.1:6006/")
        os.system('tensorboard --logdir="' + self.summary_dir + '"')
```

### 使用

```py
a = tf.constant(5.0, name="a")
b = tf.constant(6.0, name="b")
c = tf.add(a, b, name='c')

tb = TensorBoard()
with tf.Session() as sess:
    tb.save(sess.graph)
tb.board()
```
