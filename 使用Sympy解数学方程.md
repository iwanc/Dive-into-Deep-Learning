### 解线性方程的实现

- 安装Sympy：pip install Sympy
- Sympy文档：https://docs.sympy.org/latest/index.html
- 解方程组：2x-7=3,3x+7=7

```python
from sympy import *
x,y = symbols('x y')
print(solve([2 * x - y - 3, 3 * x + y - 7],[x, y]))
```

### 解微积分相关习题实现

- 公式编辑器：https://math.edrawsoft.cn/
- LaTex手写识别：https://webdemo.myscript.com/views/math/index.html
- Markdown中的LaTeX格式：https://www.jianshu.com/p/8c46e915c45e
- Mathcha -支持手写识别公式的在线数学编辑器：https://www.appinn.com/mathcha-io/
- Mathpix – 将图片数学公式转换为 LaTeX：https://www.appinn.com/mathpix/

$$
\lim _{n\rightarrow \infty }\left( \dfrac {n+3}{n+2}\right) ^{n} = E
$$

```python
from sympy import *
n = Symbol('n')
s = ((n+3)/(n+2))**n
print(limit(s, n, oo))
#result
#E
```

$$
求\int ^{n}_{0}f\left( x\right) dx，其中f\left( x\right) =\int ^{x}_{0}\dfrac {\sin t}{\pi -t}dt
$$

```python
from sympy import *
t = Symbol('t')
x = Symbol('x')
m = integrate(sin(t)/(pi-t),(t,0,x))
n = integrate(m,(x,0,pi))
print(n)
#result
#2
```

### 解微分方程

```python
from sympy import *
f = Function('f')
x = Symbol('x')
print(dsolve(diff(f(x),x) - 2*f(x)*x,f(x)))
#result
#2Eq(f(x), C1*exp(x**2)) 即f(x) = C1*exp(x**2)
```

### 矩阵化简

$$
\left( x_{1},x_{2},x_{3}\right) \times \begin{pmatrix}
a_{11} & a_{12} & a_{13} \\
a_{12} & a_{22} & a_{23} \\
a_{13} & a_{23} & a_{33}
\end{pmatrix}\times \begin{pmatrix}
x_{1} \\
x_{2} \\
x_{3}
\end{pmatrix}
$$

```python
from sympy import *
x1,x2,x3 = symbols('x1 x2 x3')
a11,a12,a13,a22,a23,a33 = symbols('a11 a12 a13 a22 a23 a33')
m = Matrix([[x1, x2, x3]])
n = Matrix([[a11, a12, a13], [a12, a22, a23], [a13, a23, a33]])
v = Matrix([[x1], [x2], [x3]])
f = m * n * v
print(f[0].subs({x1:1, x2:1, x3:1}))
#result
#a11 + 2*a12 + 2*a13 + a22 + 2*a23 + a33
```