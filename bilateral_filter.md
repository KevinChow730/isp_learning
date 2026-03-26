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
本算法采用了一种基于3D网格的实现，把二维图片的空间坐标(x,y)和其值z变成一个3D体素(x,y,z)
### 1.下采样：降低数据量
- (x,y)：空间下采样，采样步数通常设置为$\sigma_s$
- (z)：把引导值（自身像素值或边缘map）分格，采样步数通常设置为$\sigma_r$

更形式化地，设输入图像大小为 $H\times W$，像素坐标 $p=(x,y)$ 其中 $x\in[0,W-1],\;y\in[0,H-1]$。

我们选取两个采样步长：
$$
s_s\;\text{(spatial sampling)},\qquad s_r\;\text{(range sampling)}
$$
常见经验是令 $s_s\approx\sigma_s$，$s_r\approx\sigma_r$，这样在 grid 上的核标准差会落在 $\sigma_s'=\sigma_s/s_s\approx 1$、$\sigma_r'=\sigma_r/s_r\approx 1$ 的量级（核尺寸也会更小）。

#### 1.1 网格尺寸（为什么会有 floor 和 +padding）
把连续坐标 $x/s_s$、$y/s_s$ 量化到网格索引时，希望覆盖到 $x=0$ 和 $x=W-1$ 的端点。
对于一维长度 $N$，步长 $s$ 的下采样格点数常写成：
$$
n = \left\lfloor\frac{N-1}{s}\right\rfloor + 1
$$
因此空间两维分别是：
$$
n_x = \left\lfloor\frac{W-1}{s_s}\right\rfloor + 1,\qquad
n_y = \left\lfloor\frac{H-1}{s_s}\right\rfloor + 1
$$
值域（range）维度以 $E$ 的动态范围 $E_{\max}-E_{\min}$ 划分：
$$
n_z = \left\lfloor\frac{E_{\max}-E_{\min}}{s_r}\right\rfloor + 1
$$

卷积时为了避免边界效应并保证索引安全，会在每个维度加 padding（实现里取约 $2\sigma'$ 的宽度）：
$$
p_{xy}=\lfloor 2\sigma_s'\rfloor+1,\qquad p_z=\lfloor 2\sigma_r'\rfloor+1
$$
最终 grid 尺寸为：
$$
N_x = n_x + 2p_{xy},\quad N_y = n_y + 2p_{xy},\quad N_z = n_z + 2p_z
$$

#### 1.2 像素到网格的坐标映射（为什么会有 +padding+1）
对每个像素 $p=(x,y)$，先做归一化坐标：
$$
u = \frac{x}{s_s},\qquad v = \frac{y}{s_s},\qquad w = \frac{E(p)-E_{\min}}{s_r}
$$
Splat 时用最近邻量化到整数格点（round）：
$$
i=\mathrm{round}(v)+p_{xy}+1,\quad
j=\mathrm{round}(u)+p_{xy}+1,\quad
k=\mathrm{round}(w)+p_z+1
$$
其中 $+p+1$ 的作用是把有效索引整体平移到网格内部（留出 0 号边界做缓冲），从而在 blur（卷积）时邻域不会越界。

```python
downsample_width = np.floor((width - 1) / sampling_spatial) + 1 + 2 * padding_xy
downsample_height = np.floor((height - 1) / sampling_spatial) + 1 + 2 * padding_xy
downsample_depth = np.floor(edge_delta / sampling_range) + 1 + 2 * padding_z

yy, xx = np.meshgrid(np.arange(width), np.arange(height))  # 分别存储所有点的列坐标和行坐标，xx[i, j]是第i行第j列的列坐标，yy[i, j]是第i行第j列的行坐标

# 把原图像素点投影到3d网格上，网格的x、y、z分别是图像的列、行、像素值
grid_xx = np.uint16(np.round(xx / sampling_spatial) + padding_xy + 1)
grid_yy = np.uint16(np.round(yy / sampling_spatial) + padding_xy + 1)
grid_zz = np.uint16(np.round((edge - edge_min) / sampling_range) + padding_z + 1)

grid_data = np.zeros((downsample_width, downsample_height, downsample_depth), dtype=np.float32)
grid_weight = np.zeros((downsample_width, downsample_height, downsample_depth), dtype=np.float32)
```
### 2. 空间映射+3D卷积

    在原图像域里，双边滤波对每个中心像素(p)的权重都依赖于E(p)(range核中心在E(p))，所以可以看成“每个像素都有自己的一套值域核/联合核”，难以用一次固定卷积复用，lift到(x,y,z)之后，z轴把“值域相近”变成了“坐标相近”，于是“空间核×值域核”的联合权重就变成了3D空间里的一个平移不变(shift-invariant)的固定高斯核，可以用一次3D卷积在grid上统一完成blur

如果我们把引导图记作 $E$（可以取输入本身，也可以取 edge map / 亮度通道等），把像素位置记作 $p=(x,y)$，那么双边滤波的权重依赖于
$$
\|p-q\| \quad \text{以及} \quad |E(p)-E(q)|
$$
这导致对每个中心像素 $p$，值域核都“以 $E(p)$ 为中心”，是**空间变化**的。

bilateral grid 的关键是做一个“升维/坐标化”(lift)：把每个像素映射到 3D 坐标
$$
\phi(p) = \Big(\tfrac{x}{s_s}, \tfrac{y}{s_s}, \tfrac{E(p)-E_{\min}}{s_r}\Big)
$$
其中 $s_s$ 是空间采样步长（通常取 $\sigma_s$ 同量级），$s_r$ 是值域采样步长（通常取 $\sigma_r$ 同量级）。这样一来：
- **“值域相近”就变成了 “$z$ 轴坐标相近”；**
- 原来依赖 $E(p)$ 的空间变化权重，被近似成了 3D 空间中的**平移不变固定核**。

#### 2.1 Splat：把像素散射到 3D 网格（sum + count）
把连续坐标 $\phi(p)$ 量化到最近的整数网格坐标 $\pi(p)$（实现里用 `round` + padding）：
$$
\pi(p) = (i,j,k) \in \mathbb{Z}^3
$$
然后累加两张网格：
$$
\mathrm{data}(i,j,k) \;\mathrel{+}=\; I(p), \qquad \mathrm{weight}(i,j,k) \;\mathrel{+}=\; 1
$$
这里 $\mathrm{data}$ 是 data grid 的累加和（分子），$\mathrm{weight}$ 是样本数/权重和（分母）。

> 注意：此时 $\mathrm{data}$ 不是“信号值”，而是“桶里的求和”，并且网格是稀疏的（很多 cell 的 $\mathrm{weight}=0$）

```python
for i in range(width):
    for j in range(height):
        x = grid_xx[i, j]
        y = grid_yy[i, j]
        z = grid_zz[i, j]

        grid_data[x, y, z] += self.data[i, j]  # 多个像素被投影到同一个网格上，网格内的值是所有像素值的平均值
        grid_weight[x, y, z] += 1.0  # 统计每个网格上有多少像素被投影到这个网格上
```

#### 2.2 Blur：在 3D 网格上做固定高斯卷积
在 3D 空间中构造一个联合高斯核（空间 + 值域）：
$$
K(\Delta i,\Delta j,\Delta k)
= \exp\Big(-\tfrac12\big(\tfrac{\Delta i^2+\Delta j^2}{\sigma_s'^2} + \tfrac{\Delta k^2}{\sigma_r'^2}\big)\Big)
$$
其中 $\sigma_s' = \sigma_s/s_s,\; \sigma_r' = \sigma_r/s_r$ 是“以网格为单位”的标准差。
```python
kernel_width = 2. * derived_sigma_spatial + 1.
kernel_height = kernel_width
kernel_depth = 2. * derived_sigma_range + 1.

kernel_x, kernel_y, kernel_z = np.meshgrid(np.arange(kernel_width), np.arange(kernel_height), np.arange(kernel_depth))
kernel_x -= np.floor(kernel_width / 2.)
kernel_y -= np.floor(kernel_height / 2.)
kernel_z -= np.floor(kernel_depth / 2.)
kernel_r_squared = (kernel_x**2 + kernel_y**2) / derived_sigma_spatial**2\
            + (kernel_z**2) / derived_sigma_range**2
kernel = np.exp(-0.5 * kernel_r_squared)
```
然后对 $\mathrm{data}$ 和 $\mathrm{weight}$ 都做同一个 3D 卷积
$$
\overline{\mathrm{data}} = \mathrm{data} * K, \qquad \overline{\mathrm{weight}} = \mathrm{weight} * K
$$
```python
blurred_grid_data = ndimage.convolve(grid_data, kernel, mode='reflect')
blurred_grid_weight = ndimage.convolve(grid_weight, kernel, mode='reflect')
```
#### 2.3 Normalize（divide）：为什么要除、为什么 W 也要卷积
```python
# divide
blurred_grid_weight = np.asarray(blurred_grid_weight)
mask = blurred_grid_weight == 0
blurred_grid_weight[mask] = -2.  # 处理0
normalized_blurred_grid = blurred_grid_data/blurred_grid_weight
mask = blurred_grid_weight < -1
normalized_blurred_grid[mask] = 0.
```
真正想要的是局部加权**平均**，不是加权求和。因此要做归一化：
$$
\widetilde{\mathrm{data}}(i,j,k) = \frac{\overline{\mathrm{data}}(i,j,k)}{\overline{\mathrm{weight}}(i,j,k)}
$$

这一点等价于传统双边滤波的“分子/分母”结构：
$$
I'(p)=\frac{\sum_q I(q)\,w(p,q)}{\sum_q w(p,q)}
$$

为什么必须对 $\mathrm{weight}$ 也做高斯（而不是只对 $\mathrm{data}$ 卷积）？
- 因为 splat 后的网格是“sum + 稀疏采样密度”，只 blur $\mathrm{data}$ 会把“强度×密度”一起平滑，输出会随着附近落点多少而漂移。
- blur $\mathrm{weight}$ 再相除，本质上是在做 **normalized convolution**：
$$
\widetilde{\mathrm{data}} = \frac{(\mathrm{data}) * K}{(\mathrm{weight}) * K}
$$
它能正确处理空洞（$\mathrm{weight}=0$）、不均匀采样密度、以及边界处有效邻域权重和变化。

为什么不需要把核归一化（不做 $K\leftarrow K/\sum K$）？
- 因为整体尺度会在相除时抵消：把 $K$ 乘一个常数 $c$，$\overline{\mathrm{data}}$ 和 $\overline{\mathrm{weight}}$ 都会乘 $c$，最终 $\widetilde{\mathrm{data}}$ 不变。
- 而且即使做了核归一化，也无法解决“局部缺样本/空洞”问题，因为并不总是所有样本都参与贡献，真正的有效分母应该是 $(\mathrm{weight}*K)$，而不是常数 $\sum K$。

由此，我们得到了3D网格中完成的保边平滑结果

### 3. 上采样
上一步得到的是定义在 3D 网格上的连续函数 $\widetilde{\mathrm{data}}(i,j,k)$。接下来要把它取回每个像素的位置（slice）：

对每个像素 $p$，我们计算其连续 3D 坐标：
$$
\phi(p) = \Big(\tfrac{x}{s_s}+\text{pad}, \tfrac{y}{s_s}+\text{pad}, \tfrac{E(p)-E_{\min}}{s_r}+\text{pad}\Big)
$$
然后对 $\widetilde{\mathrm{data}}$ 做三线性插值：
$$
I'(p) = \mathrm{trilinear\_interp}(\widetilde{\mathrm{data}},\; \phi(p))
$$
```python
# upsample
grid_xx_r = (xx / sampling_spatial) + padding_xy + 1.
grid_yy_r = (yy / sampling_spatial) + padding_xy + 1.
grid_zz_r = (edge - edge_min) / sampling_range + padding_z + 1.

n_x, n_y, n_z = normalized_blurred_grid.shape  # 网格中有多少离散采样点
points = (np.arange(n_x), np.arange(n_y), np.arange(n_z))

x_i = (grid_xx_r, grid_yy_r, grid_zz_r)

output = interpolate.interpn(points, 
                            normalized_blurred_grid, 
                            x_i, 
                            method="linear")
```

