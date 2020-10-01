<link rel="stylesheet" href="https://zhmhbest.gitee.io/hellomathematics/style/index.css">
<script src="https://zhmhbest.gitee.io/hellomathematics/style/index.js"></script>

# [Scikit-learn](../index.html)

[TOC]

## 特征工程

### [`特征抽取`](src/feature_extraction.py)

原始数据转化为更好地代表预测模型的潜在问题的特征的过程。

#### TF-IDF

```txt
tf : 词频（term frequency ）
idf: 逆文档频率（inverse document frequency）
```

用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。

### [`特征过滤`](src/feature_filter.py)

#### 主成分分析

利用正交变换把由现行相关变量表示的观测数据转化为少数几个由线性无关变量表示的数据，线性无关的变量称为主成分。主成分的个数通常小于原始变量的个数，所以属于降维方法。主成分分析虽然损失了少量信息，但提高了有效信息的比率。

### [`特征预处理`](src/feature_preprocessing.py)

#### OneHot

离散特征的取值之间没有大小的意义，OneHot编码使用$n$位状态寄存器来对$n$个状态进行编码，每个状态都由他独立的寄存器位，并且在任意时候，其中只有一位有效。

#### 归一化

$$x =
    \left(\dfrac{x - x_{min}}{x_{max} - x_{min}} \right)
    \times
    (r_{max} - r_{min})
    +
    r_{min}
$$

- $r_{min}$：目标区间最小值，一般为$0$。
- $r_{max}$：目标区间最大值，一般为$1$。

归一化是一种无量纲处理手段，使物理系统数值的绝对值变成某种相对值关系。简化计算，缩小量值的有效办法。

**缺陷**：归一化易受异常点影响。

#### 标准化

$$x = \dfrac{x - x_{mean}}{x_σ}$$

变换到均值为0，方差为1的范围内。数据量充足时，受异常点影响较小。

#### 正则化

$$L_1(x) = \dfrac{x}{ \sum\limits_{j\ of\ cols} |x_j| }$$

$$L_2(x) = \dfrac{x}{ \sqrt{ \sum\limits_{j\ of\ cols} x_j^2 } }$$

## 常用内容

>[Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

```py
# 线性回归
from sklearn.linear_model import LinearRegression

# 岭回归（带正则化）
from sklearn.linear_model import Ridge

# 逻辑回归（二分类问题）
from sklearn.linear_model import LogisticRegression

# 带正则化的随机梯度下降法线性回归
from sklearn.linear_model import SGDRegressor

# 朴素贝叶斯多分类器
from sklearn.naive_bayes import MultinomialNB

# 决策树
from sklearn.tree import DecisionTreeClassifier

# 随机森林
from sklearn.ensemble import RandomForestClassifier

# K-近邻
from sklearn.neighbors import KNeighborsClassifier

# 网格搜索
from sklearn.model_selection import GridSearchCV

# K-Means聚类
from sklearn.cluster import KMeans

# 模型评价方法
from sklearn.metrics import *
```
