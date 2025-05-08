# Triple Product for Quasisymmetry

不需要指定对称性方向的 准对称性 优化目标 $f_T$ 被称为 Triple Product, 形式为：

$$
f_T = \nabla\psi\times\nabla B\cdot \nabla(\mathbf{B}\cdot\nabla B) \quad [T^4/m^2]
$$

其无量纲的形式为

$$
f_{T,norm} = \frac{L_{\text{reference}}^2}{B_{\text{reference}}^4}f_T
\quad \text{or}\quad
\frac{R_0^2}{B_{0}^4}f_T
$$

## On the Magnetic Surface

在连续嵌套磁面假设中，磁面上的磁场强度可以表述为

$$
\mathbf{B} = \nabla\psi\times\nabla\theta+\iota\nabla\phi\times\nabla\psi =B^s\nabla s+B^\theta\nabla\theta+B^\phi\nabla\phi
$$

由于磁场与磁面相切，则 $\mathbf{B}\cdot\nabla s = B^s = 0$

任意标量 $f(s,\theta,\phi)$ 的梯度为 $\nabla f = \partial_s f\nabla s + \partial_\theta f\nabla \theta + \partial_\phi f\nabla \phi$ , 则 $\mathbf{B}\cdot\nabla B$ 可以表示为：

$$
\begin{aligned}
\mathbf{B}\cdot\nabla B &= (B^\theta\nabla\theta+B^\phi\nabla\phi)\cdot
(\partial_s B\nabla s + \partial_\theta B\nabla \theta + \partial_\phi B\nabla \phi) \\
&= B^\theta\partial_\theta B + B^\phi\partial_\phi B \rightarrow g
\end{aligned}
$$

进一步的对标量 $\mathbf{B}\cdot\nabla B$ 进行梯度运算：

$$
\begin{aligned}
\nabla(\mathbf{B}\cdot\nabla B) &= 
\{(\partial_\theta B^\theta)\partial_\theta B +B^\theta\partial_{\theta\theta}B  + 
(\partial_\theta B^\phi)\partial_\phi B +B^\phi\partial_{\theta\phi}B\}\cdot \nabla\theta \quad\rightarrow g_\theta\cdot \nabla\theta \\
&+ 
\{(\partial_\phi B^\theta)\partial_\theta B +B^\theta\partial_{\theta\phi}B  + 
(\partial_\phi B^\phi)\partial_\theta B +B^\phi\partial_{\phi\phi}B\}\cdot\nabla\phi \quad\rightarrow g_\phi\cdot \nabla\phi
\end{aligned}
$$

对于项 $\nabla\psi\times\nabla B$ ：

$$
\begin{aligned}
\nabla\psi\times\nabla B \cdot\nabla(\vec{B}\cdot\nabla B)= &-\frac{\Phi_{edge}}{2\pi}\nabla s \times (\partial_\theta B\nabla \theta + \partial_\phi B\nabla \phi) \\
&\cdot\{(\partial_\theta B^\theta)\partial_\theta B +B^\theta\partial_{\theta\theta}B\  + 
(\partial_\theta B^\phi)\partial_\phi B +B^\phi\partial_{\theta\phi}B \}\cdot\nabla\theta
\\&+ \{(\partial_\phi B^\theta)\partial_\theta B +B^\theta\partial_{\theta\phi}B  + 
(\partial_\phi B^\phi)\partial_\theta B +B^\phi\partial_{\phi\phi}B \}\cdot\nabla\phi \\
=& -\frac{\Phi_{edge}}{2\pi}\nabla s \times (\partial_\theta B\nabla \theta + \partial_\phi B\nabla \phi) \cdot(g_\theta\cdot\nabla\theta+g_\phi\cdot\nabla\phi) \\
=&-\frac{\Phi_{edge}}{2\pi\sqrt{g}} (g_\phi\partial_\theta B - g_\theta\partial_\phi B) 
\end{aligned}
$$

这里应用了 $\nabla s\times\nabla\theta\cdot\nabla\phi = 1/\sqrt{g}$

# In VMEC

在VMEC中，磁场通过傅里叶系数表示（这里只考虑仿星器对称）

$$
B(\theta,\phi) = \sum B_{m,n}\cos(m\theta-n\phi)
$$

因此高阶导数可以直接通过傅里叶系数得到，而避免使用数值梯度产生的误差。

$$
\begin{aligned}
\partial_\theta B &= -\sum mB_{mn}\sin(m\theta-n\phi) \\
\partial_\phi B &=\quad \sum nB_{mn}\sin(m\theta-n\phi) \\
\partial_{\theta\theta} B &= -\sum m^2B_{mn}\cos(m\theta-n\phi)\\
\partial_{\phi\phi} B &= -\sum n^2B_{mn}\cos(m\theta-n\phi)\\
\partial_{\theta\phi} B &= \quad\sum mnB_{mn}\cos(m\theta-n\phi)
\end{aligned}
$$

在VMEC的输出文件中同样保存了 $B^\theta$ 和 $B^\phi$ 的傅里叶系数。例如`bsupumnc/bsupvmnc`。 与上述方法完全一致，可以以傅里叶系数的方式较为解析的获得各阶分量。

## 无量纲化

准对称目标 $f_T$ 的单位为 $[T^4/m^2]$ ， 因此需要多种方式无量纲化该量。

**GX-like**

一些湍流输运的代码中通过以下量无量纲化：

$$
\begin{aligned}
L_{\text{reference}} &= \text{Aminorp} \\
B_{\text{reference}} &= 2 \cdot (-\frac{\Phi_\text{edge}}{2\pi})/L_{\text{reference}} ^2
\end{aligned}
$$

**DESC**

DESC中通过大半径进行无量纲化

$$
\begin{aligned}
L &= \text{Rmajorp} \\
B &=\frac{\text{mean}(|B|\cdot\sqrt{g})}{\text{mean}(\sqrt{g})}
\end{aligned}
$$

## 磁面平均

任意一个量的磁面平均可以写为

$$
\left< A \right> = \frac{\int_0^{2\pi}\int_0^{2\pi}A\sqrt{g}d\theta d\phi}{\int_0^{2\pi}\int_0^{2\pi}\sqrt{g}d\theta d\phi}
$$


