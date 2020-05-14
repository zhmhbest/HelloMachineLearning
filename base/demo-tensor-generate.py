import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
