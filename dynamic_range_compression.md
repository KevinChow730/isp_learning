# ISP DRC 基底分量线性压缩算法
## 核心设计逻辑
我们需要对**双边滤波得到的基底亮度**做一段**线性映射压缩**，目标：
1. 用**简单线性变换**完成基底动态范围压缩
2. 变换必须经过**指定的控制点 (C, T)**，保证压缩曲线可控
3. 以图像中点为中心对称压缩，不偏移整体亮度
4. 由约束条件**唯一解算压缩系数 F**，而非手动设定经验值

---

## 变量定义
| 代码变量 | 数学符号 | 含义 |
|---------|----------|------|
| clip_range[1] | $L_{max}$ | 图像亮度最大值（动态范围上限） |
| drc_bound[0] | $k_0$ | 低亮控制点参数 |
| drc_bound[1] | $k_1$ | 高亮控制点参数 |
| - | $C$ | 映射输出定点 |
| - | $T$ | 映射输入定点 |
| y_bilateral_filtered | $Y_{in}$ | 输入基底亮度 |
| - | $F$ | 线性压缩系数（斜率） |
| y_bilateral_filtered_contrast_reduced | $Y_{out}$ | 输出压缩后基底 |

---

## 计算真实范围的定点 (C, T)
将 0~255 的参数缩放到图像实际亮度范围：
$$
C = k_0 \cdot \frac{L_{max}}{255}
$$
$$
T = k_1 \cdot \frac{L_{max}}{255}
$$
这两个值构成线性映射的**约束定点：输入 T → 输出 C**。

---

## 线性映射公式（中心化对称压缩）
为了不偏移图像中点，使用**中点中心化线性变换**：
$$
Y_{out} = F \cdot \left(Y_{in} - \frac{L_{max}}{2}\right) + \frac{L_{max}}{2}
$$

### 公式含义
1. $Y_{in} - L_{max}/2$：将亮度以**中点为中心对齐**
2. $F \cdot (\dots)$：用系数 F 进行**线性缩放（压缩）**
3. $+ L_{max}/2$：恢复中心点，保证整体亮度不漂移

这就是我们要使用的**线性映射模型**。

---

## 五、代入定点 (T, C) 解算 F
### 约束条件
当输入亮度为 $T$ 时，输出必须等于 $C$：
$$
Y_{in}=T \quad \Rightarrow \quad Y_{out}=C
$$

### 代入公式解方程
$$
C = F \cdot \left(T - \frac{L_{max}}{2}\right) + \frac{L_{max}}{2}
$$

移项求解 F：
$$
F \cdot \left(T - \frac{L_{max}}{2}\right) = C - \frac{L_{max}}{2}
$$

$$
F = \frac{C - \cfrac{L_{max}}{2}}{T - \cfrac{L_{max}}{2}}
$$

### 工程实现
### 1. 由定点约束解出的原始公式
$$
F = \frac{C - \frac{L_{max}}{2}}{T - \frac{L_{max}}{2}} = \frac{2C - L_{max}}{2T - L_{max}}
$$

### 2. 结合代码中 C、T 的定义（比例关系）
在本算法中，控制点满足：
$$
C = k_0 \cdot \frac{L_{max}}{255}, \quad T = k_1 \cdot \frac{L_{max}}{255}
$$
因此存在固定等价关系：
$$
2C-L_{max} = -\frac{T}{L_{max}}(C+L_{max}), \quad 2T-L_{max} = -(T-C)
$$

### 3. 代入化简得到工程实现形式
$$
F = \frac{ -\frac{T}{L_{max}}(C+L_{max}) }{ -(T-C) } = \frac{T \cdot (C + L_{max})}{L_{max} \cdot (T - C)}
$$
```python
C = clip_range[1] * drc_bound[0] / 255.
T = clip_range[1] * drc_bound[1] / 255.
F = (T * (C + clip_range[1])) / (clip_range[1] * (T - C))
base_compressed = (F * (y_bilateral - clip_range[1]/2)) + clip_range[1]/2
```

