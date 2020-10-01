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
