# autograd

- $x = \left[\begin{array}{c} x_1 & x_2 \\ x_3 & x_4 \end{array}\right] =\left[\begin{array}{c} 1 & 1 \\ 1 & 1 \end{array}\right]$
- $y = 3(x+2)^2$
- $o = \dfrac{1}{4} \sum\limits_{i} y_{i}$
- $\dfrac{∂o}{∂x_i} = \dfrac{3}{2}(x_i+2)$；$\dfrac{∂o}{∂x_i} \bold{|}_{x_i=1} = \dfrac{9}{2}$
