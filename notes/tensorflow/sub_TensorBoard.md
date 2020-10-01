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
