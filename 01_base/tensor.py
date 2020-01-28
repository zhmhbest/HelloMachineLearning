import tensorflow as tf


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
    填充
========================================""".strip())

# 产生以给定值填充的张量
tensor_fill = tf.fill([2, 3], 99)
print(tensor_fill.name)

# 产生以0填充的张量
tensor_zeros = tf.zeros([2, 3], tf.float32, name=None)
print(tensor_zeros.name)

# 产生以1填充的张量
tensor_ones = tf.ones([2, 3], tf.float32, name=None)
print(tensor_ones.name)


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
