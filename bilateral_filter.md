# 双边滤波
双边滤波（Bilateral Filter）是一种保边去噪的非线性滤波，核心是同时考虑空间距离与像素值差异，用两个高斯核乘积做加权平均。能在保留边缘detail的情况下平滑图片base
## 核心公式
### 1. 滤波输出
$$
I'(p) = \frac{1}{W_p} \sum_{q \in \Omega} I(q) \cdot w_s(p,q) \cdot w_r(p,q)
$$
其中归一化系数：$W_p = \sum_{q \in \Omega} w_s(p,q) \cdot w_r(p,q)$，滤波窗口直径 $d$，空间标准差 $\sigma_s$，值域标准差 $\sigma_r$

### 2. 空间权重
$$
w_s(p,q) = \exp\left(-\frac{\|p - q\|^2}{2\sigma_s^2}\right)
$$

### 3. 值域权重
$$
w_r(p,q) = \exp\left(-\frac{|I(p)-I(q)|^2}{2\sigma_r^2}\right)
$$

### 4. 联合权重
$$
w(p,q) = w_s(p,q) \cdot w_r(p,q)
$$

传统双边滤波计算量大，值域权重因每个像素及其邻域不同而不同，难以实现

## J Chen, et al. Real-time Edge-Aware Image Processing with the Bilateral Grid
本算法采用了一种基于3D网格的实现
