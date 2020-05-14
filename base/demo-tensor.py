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
