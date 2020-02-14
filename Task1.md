<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

## 1线性回归
### 1.1例子
  $$price = w_{area}.area + w_{age}.age +b$$
  这是一个简单的房价预测，房价和房子的面积和年龄有线性关系。

通用的线性回归模型我们使用$y = w\times x+b$来表示
### 1.2损失函数
平方损失函数$l_i(w,b) = \frac{1}{2}(\hat{y}^{(i)}-y^{(i)})^2$

### 1.3随机梯度下降
在求数值解的优化算法中，小批量随机梯度下降（mini-batch stochastic gradient descent）在深度学习中被广泛使用。它的算法很简单：先选取一组模型参数的初始值，如随机选取；接下来对参数进行多次迭代，使每次迭代都可能降低损失函数的值。在每次迭代中，先随机均匀采样一个由固定数目训练数据样本所组成的小批量（mini-batch），然后求小批量中数据样本的平均损失有关模型参数的导数（梯度），最后用此结果与预先设定的一个正数的乘积作为模型参数在本次迭代的减小量。

$$(w,b) \leftarrow(w,b) - \frac{\eta}{|B|}\sum_{i\in B}\delta_{(w,b)}l^{(i)}(w,b)	$$

#### 随机梯度下降从0开始代码例子分析
随机梯度下降函数自定义
```python
def sgd(params, lr, batch_size): 
    for param in params:
        param.data -= lr * param.grad / batch_size
```
其中lr是我们训练中设置的系数决定每次步长的大小。相当于我们前面公式中的$\eta参数$

```python
or epoch in range(num_epochs):  # training repeats num_epochs times
    # in each epoch, all the samples in dataset will be used once
    
    # X is the feature and y is the label of a batch sample
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()  
        # calculate the gradient of batch sample loss 
        l.backward()  
        # using small batch random gradient descent to iter model parameters
        sgd([w, b], lr, batch_size)  
        # reset parameter gradient
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))
```
代码中应用了.grad方法该方法能够反向求出梯度，方便我们理解随机梯度下降的原理。

顺便耶记录一下pytorch中已经内置的线性回归代码
```python
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()      # call father function to init 
        self.linear = nn.Linear(n_feature, 1)  # function prototype: `torch.nn.Linear(in_features, out_features, bias=True)`

    def forward(self, x):
        y = self.linear(x)
        return y
    
net = LinearNet(num_inputs)
print(net)
```

### 1.4课后错题分析
假如你正在实现一个全连接层，全连接层的输入形状是$7 \times 8$ ，输出形状是$7 \times 1$，其中7是批量大小，则权重参数ww和偏置参数bb的形状分别是____和____

答案：$8\times 1$，$1 \times 1$
b是是一个常数每个实例都是一样的所以是$1\times 1$的。

## 2.softmax
### 2.1基本概念
主要用来处理离散的分类问题。（ps 离散问题和连续问题差异：比如说预测房价房价是一个连续的数。所谓的离散问题就是比如说判断是否属于这一类。）

𝑠𝑜𝑓𝑡𝑚𝑎𝑥回归是一个单层神经网络。

softmax公式具体[链接](https://baike.baidu.com/item/softmax%20%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92/22689563?fr=aladdin)如下
注意的是softmax中b参数取决于分类的种类数量，比如分三类就有三个，和前面的线性回归有些不同。

## 3.多层感知机
### 3.1 概念
简单的理解就是模型的层数便多了。并且嵌入了激活函数。
### 3.2 激活函数
引入激活函数的原因：一直使用嵌套的线性函数训练出的效果和单一曾的线性模型效果差不多

激活函数有很多种如relu常用的，sigmoid，还有tanh
如何选择激活函数：在中间层可以使用relu等(计算快，而且是一个大的区间，不会出现梯度消失的问题)。在输出层可以使用sigmoid来将值转化到0-1之间。
