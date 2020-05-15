import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
0阶张量：标量（Scalar）也就是1个实数
1阶张量：向量（Vector）也就是1维数组
2阶张量：矩阵（Matrix）也就是2维数组
n阶张量：n维数组
"""

# 定义一个常数tensor(张量)
tensor_constant = tf.constant([
    [1.1, 2.2, 3.3],
    [4.4, 5.5, 6.6]
], name='a', dtype=tf.float32)


print("""
========================================
    OP(Operation, 运算)
    节点在图中被称为OP，OP即某种抽象计算。
========================================""".strip())
print(tensor_constant.op)


print("""
========================================
    张量的属性
========================================""".strip())
print("name", tensor_constant.name)
print("type", tensor_constant.dtype)
print("shape", tensor_constant.shape)
print("graph", tensor_constant.graph)


print("""
========================================
    调整
========================================""".strip())

# 类型转换
result_cast = tf.cast(tensor_constant, tf.int32, name='result_cast')
print(tensor_constant.dtype, '=>', result_cast.dtype)

# 结构调整
result_reshape = tf.reshape(tensor_constant, [3, 2], name='result_reshape')
print(tensor_constant.shape, '=>', result_reshape.shape)


print("""
========================================
    序列张量
========================================""".strip())
# 从1开始步长为3不超过10的序列
tensor_range = tf.range(1, 10, 3, dtype=None, name=None)
print(tensor_range)

# 10~100等分为5份
tensor_space = tf.linspace(10.0, 100.0, 5, name=None)
print(tensor_space)


print("""
========================================
    填充张量
========================================""".strip())
# 产生以给定值填充的张量
tensor_fill = tf.fill([2, 3], 99, name=None)
print(tensor_fill)

# 产生以0填充的张量
tensor_zeros = tf.zeros([2, 3], tf.float32, name=None)
print(tensor_zeros)

# 产生以1填充的张量
tensor_ones = tf.ones([2, 3], tf.float32, name=None)
print(tensor_ones)

# 产生对角线为[1, 2, 3, 4]其余为0的二维张量
tensor_diag = tf.diag([1, 2, 3, 4], name=None)
print(tensor_diag)


print("""
========================================
    随机数
========================================""".strip())

# 正态分布随机数(shape, mean:平均数, stddev:标准差)
tensor_random1 = tf.random_normal([2, 3], 10, 0.6, name=None)
print(tensor_random1)

# 正态分布随机数，偏离2个标准差的随机值会被重新生成
tensor_random2 = tf.truncated_normal([2, 3], 10, 0.6, name=None)
print(tensor_random2)

# 均匀分布随机数(shape, min, max)
tensor_random3 = tf.random_uniform([2, 3], 1, 10, name=None)
print(tensor_random3)

# Γ(Gamma)随机数(shape, alpha, beta)
# tensor_random4 = tf.random_gamma(...)
# print(tensor_random4)
