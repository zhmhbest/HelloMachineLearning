<link rel="stylesheet" href="https://zhmhbest.gitee.io/hellomathematics/style/index.css">
<script src="https://zhmhbest.gitee.io/hellomathematics/style/index.js"></script>

# [Keras](../index.html)

[TOC]

## Hello

>[Keras API](https://keras.io/api/)

```flow
st=>start: 开始

data=>inputoutput: 数据载入
preprocessing=>operation: 数据预处理

model=>operation: 模型定义
compile=>operation: 模型编译
training=>operation: 模型训练
assessment=>operation: 模型评估

predict=>operation: 预测
save=>operation: 保存模型
ed=>end: 结束

st->data->preprocessing->model->compile->training->assessment->predict->save->ed
```

@import "./src/demo-hello.py"

<!-- >[`demo-hello.py`](./src/demo-hello.py)
 -->
