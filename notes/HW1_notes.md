# HW1 百科全书式笔记：ndarray + pandas

> 对应文件：`tutorial/HW1.ipynb`
> 覆盖范围：Python 基础语法 → pandas 数据处理 → PyTorch 张量操作

---

## 目录

- [一、Python 基础语法速查](#一python-基础语法速查)
- [二、import 导入机制](#二import-导入机制)
- [三、os 模块：文件与路径操作](#三os-模块文件与路径操作)
- [四、StringIO：内存中的文件对象](#四stringio内存中的文件对象)
- [五、pandas 核心操作](#五pandas-核心操作)
- [六、PyTorch 张量（Tensor）核心操作](#六pytorch-张量tensor核心操作)
- [七、HW1 逐题解析](#七hw1-逐题解析)

---

## 一、Python 基础语法速查

### 1.1 变量与赋值

```python
x = 10          # 整数
y = 3.14        # 浮点数
s = "hello"     # 字符串
flag = True     # 布尔值
```

Python 是**动态类型**语言——变量不需要声明类型，赋值时自动推断。

### 1.2 三引号字符串

```python
csv_data = """NumRooms,Alley,Price
NA,Pave,127500
2,NA,106000
"""
```

| 语法 | 含义 |
|------|------|
| `"""..."""` | 多行字符串，保留换行符 |
| `"..."` | 单行字符串 |
| `\n` | 字符串中的换行转义 |

三引号字符串中的每一个换行都是真实的 `\n` 字符，常用于嵌入多行文本数据。

### 1.3 f-string 与 print

```python
name = "tensor"
print(f"this is a {name}")   # f-string 格式化
print("a:", a, "b:", b)       # 逗号分隔，自动加空格
```

### 1.4 切片语法 `[start:stop:step]`

```python
lst = [0, 1, 2, 3, 4]
lst[:2]    # [0, 1]        → 从头取到索引 2（不含）
lst[-1]    # 4             → 最后一个元素
lst[1:4]   # [1, 2, 3]     → 索引 1 到 4（不含）
lst[::2]   # [0, 2, 4]     → 每隔 2 个取一个
```

此切片语法在 `list`、`numpy`、`pandas`、`torch` 中**通用**。

### 1.5 `id()` 函数

```python
id(X)  # 返回对象在内存中的唯一标识（地址）
```

用途：判断两次操作后变量是否**还是同一个对象**（而不是创建了新对象）。

### 1.6 `type()` 函数

```python
type(3.5)          # <class 'float'>
type(torch.tensor([1]))  # <class 'torch.Tensor'>
```

返回对象的类型/类名。

---

## 二、import 导入机制

### 2.1 基本导入方式

```python
import torch                  # 导入整个模块，使用时 torch.xxx
import pandas as pd           # 导入并起别名，使用时 pd.xxx
from io import StringIO       # 从模块中导入特定对象，直接用 StringIO
import os                     # 导入标准库模块
```

| 写法 | 调用方式 | 场景 |
|------|----------|------|
| `import torch` | `torch.tensor(...)` | 完整导入 |
| `import pandas as pd` | `pd.read_csv(...)` | 起别名，减少打字 |
| `from io import StringIO` | `StringIO(...)` | 只需要模块中的某个类/函数 |

### 2.2 HW1 用到的模块一览

| 模块 | 作用 |
|------|------|
| `torch` | PyTorch 深度学习框架，提供张量运算、自动求导、GPU 加速 |
| `pandas` (别名 `pd`) | 数据分析库，提供 DataFrame 表格结构，擅长读 CSV、清洗数据 |
| `io.StringIO` | 把字符串包装成"文件对象"，让 `pd.read_csv` 可以读字符串 |
| `os` | 操作系统接口——创建目录、拼接路径、文件操作 |

### 2.3 `torch.manual_seed(0)`

```python
torch.manual_seed(0)
```

- 设置 PyTorch 的**随机种子**为 0
- 目的：让随机操作（如 `torch.randn`）每次产生**相同的结果**，保证实验可复现
- `manual_seed` = "手动设定种子"

---

## 三、os 模块：文件与路径操作

### 3.1 `os.path.join()`

```python
os.path.join('..', 'data', 'house_tiny.csv')
# Windows 结果: '..\data\house_tiny.csv'
# Linux 结果:   '../data/house_tiny.csv'
```

- 跨平台拼接路径，自动处理 `/` 和 `\` 的差异
- `'..'` 表示上一级目录

### 3.2 `os.makedirs()`

```python
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
```

| 参数 | 含义 |
|------|------|
| 第一个参数 | 要创建的目录路径 |
| `exist_ok=True` | 如果目录已存在，不报错（默认 `False` 会报 `FileExistsError`） |

递归创建目录，类似命令行的 `mkdir -p`。

### 3.3 `open()` + `with` 语句

```python
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')
```

| 要素 | 含义 |
|------|------|
| `with ... as f:` | 上下文管理器，代码块结束后自动关闭文件 |
| `'w'` | 写模式（覆盖），其他常见：`'r'` 读、`'a'` 追加 |
| `f.write(...)` | 向文件写入字符串 |
| `\n` | 换行符 |

---

## 四、StringIO：内存中的文件对象

```python
from io import StringIO

csv_data = """NumRooms,Alley,Price
NA,Pave,127500
"""

data = pd.read_csv(StringIO(csv_data))
```

- `pd.read_csv()` 的第一个参数需要是**文件路径**或**文件对象**
- `StringIO(csv_data)` 把一段字符串包装成一个"假文件"，让 `read_csv` 以为在读文件
- 好处：不用真的在磁盘上创建 CSV 文件，适合演示和测试

---

## 五、pandas 核心操作

### 5.1 核心数据结构

| 结构 | 维度 | 类比 |
|------|------|------|
| `Series` | 一维 | 一列数据（带索引的数组） |
| `DataFrame` | 二维 | 一张表格（行×列） |

```python
# DataFrame 示例
#    NumRooms Alley   Price
# 0       NaN  Pave  127500
# 1       2.0   NaN  106000
# 2       4.0   NaN  178100
# 3       NaN   NaN  140000
```

- 每一列是一个 `Series`
- `NaN` (Not a Number) 表示缺失值
- 行索引默认是 0, 1, 2, ...

### 5.2 `pd.read_csv()`

```python
data = pd.read_csv(StringIO(csv_data))
data = pd.read_csv('path/to/file.csv')
```

- 读取 CSV 格式数据，返回 `DataFrame`
- 自动将第一行识别为**列名**
- `NA`、空白单元格自动识别为 `NaN`

### 5.3 `iloc` — 按位置索引（integer-location）

```python
data.iloc[:, :2]    # 所有行，前 2 列
data.iloc[:, -1]    # 所有行，最后 1 列
data.iloc[0, 1]     # 第 0 行，第 1 列（单个值）
data.iloc[1:3, :]   # 第 1~2 行，所有列
```

**语法：`data.iloc[行切片, 列切片]`**

| 符号 | 含义 |
|------|------|
| `:` | 全部 |
| `:2` | 从头到索引 2（不含） |
| `-1` | 最后一个 |
| `1:3` | 索引 1 到 3（不含） |

**对比 `loc` 与 `iloc`：**

| 方法 | 索引方式 | 示例 |
|------|----------|------|
| `iloc` | **整数位置**（0, 1, 2...） | `data.iloc[:, 0]` → 第 0 列 |
| `loc` | **标签名** | `data.loc[:, 'Price']` → Price 列 |

### 5.4 `select_dtypes()` — 按数据类型筛选列

```python
inputs.select_dtypes(include='number').columns
```

- `include='number'`：选出所有数值类型的列（`int64`, `float64` 等）
- `.columns`：返回这些列的**列名列表**（`Index` 对象）

| include 参数 | 匹配的类型 |
|-------------|-----------|
| `'number'` | int, float 等所有数值 |
| `'object'` | 字符串/混合类型 |
| `'bool'` | 布尔类型 |

返回的是一个新 DataFrame，只包含符合条件的列。

### 5.5 `.copy()` — 深拷贝

```python
inputs = inputs.copy()
```

- 创建 DataFrame 的**独立副本**
- 之后修改 `inputs` 不会影响原始 `data`
- 如果不 `.copy()`，`inputs` 可能只是 `data` 的**视图**（view），修改 `inputs` 会连带修改 `data`，并产生 `SettingWithCopyWarning`

### 5.6 `fillna()` — 填充缺失值

```python
inputs[num_cols] = inputs[num_cols].fillna(inputs[num_cols].mean())
```

**拆解：**

| 部分 | 含义 |
|------|------|
| `inputs[num_cols]` | 选出数值列（用列名列表索引） |
| `.mean()` | 计算每列的均值（自动跳过 NaN） |
| `.fillna(value)` | 把所有 NaN 替换为 `value` |

**fillna 常见策略：**

```python
df.fillna(0)              # 用 0 填充
df.fillna(df.mean())      # 用各列均值填充（数值列常用）
df.fillna(df.median())    # 用中位数填充
df.fillna(method='ffill') # 用前一个有效值向下填充
```

本题用**均值填充** `NumRooms` 列：`(2 + 4) / 2 = 3.0`，所以 NaN 变成 3.0。

### 5.7 `pd.get_dummies()` — One-Hot 编码

```python
inputs = pd.get_dummies(inputs, dummy_na=True)
```

**作用：把类别列（字符串）转换为数值的 0/1 列。**

转换前：

```
   NumRooms Alley
0       3.0  Pave
1       2.0   NaN
2       4.0   NaN
3       3.0   NaN
```

转换后：

```
   NumRooms  Alley_Pave  Alley_nan
0       3.0        True      False
1       2.0       False       True
2       4.0       False       True
3       3.0       False       True
```

| 参数 | 含义 |
|------|------|
| `dummy_na=True` | 把 NaN 也当作一个类别，生成 `Alley_nan` 列 |
| `dummy_na=False`（默认） | 忽略 NaN，NaN 对应行所有 dummy 列都是 0 |

**为什么需要 One-Hot？**
- 机器学习模型只能处理数值输入
- 类别 "Pave" 不能直接作为数字
- One-Hot 把 k 个类别变成 k 个 0/1 列，无引入大小关系

### 5.8 `.to_numpy()` — DataFrame → NumPy 数组

```python
inputs.to_numpy(dtype=float)
```

- 将 DataFrame 转换为 `numpy.ndarray`
- `dtype=float` 强制转为浮点数（True/False → 1.0/0.0）
- 之后可以用 `torch.tensor()` 包装成 PyTorch 张量

---

## 六、PyTorch 张量（Tensor）核心操作

### 6.1 什么是张量（Tensor）

张量是**多维数组**的统称：

| 维度 | 数学名称 | 示例形状 | Python 写法 |
|------|----------|----------|-------------|
| 0 维 | 标量 scalar | `()` | `torch.tensor(3.5)` |
| 1 维 | 向量 vector | `(3,)` | `torch.tensor([1, 2, 3])` |
| 2 维 | 矩阵 matrix | `(3, 4)` | `torch.tensor([[1,2],[3,4]])` |
| 3 维 | 3阶张量 | `(2, 3, 4)` | 图像批次等 |
| n 维 | n阶张量 | `(d1, d2, ..., dn)` | 深度学习中很常见 |

### 6.2 创建张量

```python
# 从 Python 列表创建
torch.tensor([[1, 2], [3, 4]])

# 指定数据类型
torch.tensor([1, 2], dtype=torch.float32)

# 从 numpy 数组创建
torch.tensor(np_array)
```

**常用工厂函数：**

| 函数 | 作用 | 示例 |
|------|------|------|
| `torch.arange(n)` | 生成 0 到 n-1 的整数序列 | `torch.arange(12)` → `[0,1,...,11]` |
| `torch.zeros(shape)` | 全 0 张量 | `torch.zeros((2,2))` |
| `torch.ones(shape)` | 全 1 张量 | `torch.ones((2,2))` |
| `torch.randn(shape)` | 标准正态随机 | `torch.randn((3,4))` |
| `torch.eye(n)` | n×n 单位矩阵 | `torch.eye(3)` |

### 6.3 `dtype` — 数据类型

| dtype | 说明 | 位数 |
|-------|------|------|
| `torch.float32` / `torch.float` | 单精度浮点（**默认**） | 32 bit |
| `torch.float64` / `torch.double` | 双精度浮点 | 64 bit |
| `torch.int32` / `torch.int` | 整数 | 32 bit |
| `torch.int64` / `torch.long` | 长整数（索引常用） | 64 bit |
| `torch.bool` | 布尔 | 8 bit |

`torch.tensor(data, dtype=torch.float32)` 显式指定类型。

### 6.4 `.shape` 与 `.numel()`

```python
A = torch.zeros((2, 6))
A.shape    # torch.Size([2, 6])  → 形状
A.numel()  # 12                  → 元素总数 (num elements)
```

- `.shape` 返回 `torch.Size` 对象（可当元组用）
- `.numel()` = 所有维度大小的乘积，即元素总个数

### 6.5 `reshape()` — 改变形状

```python
M = torch.arange(12)       # 形状 (12,)
A = M.reshape(2, 6)        # 形状 (2, 6)
B = M.reshape(3, -1)       # 形状 (3, 4)，-1 自动推断
C = M.reshape(-1)          # 形状 (12,)，展平
```

**核心规则：**
- reshape 前后元素总数（`numel()`）必须相同
- `-1` 表示"让 PyTorch 自动计算这个维度的大小"
- 只能有**一个** `-1`
- `reshape` 不改变元素在内存中的存储顺序，只改变"如何解读"这些元素

**reshape 的数据排列：行优先（C order）**

```
原始: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

reshape(3, 4):
  [[0,  1,  2,  3],     ← 先填满第 0 行
   [4,  5,  6,  7],     ← 再填第 1 行
   [8,  9, 10, 11]]     ← 最后第 2 行

reshape(2, 6):
  [[0,  1,  2,  3,  4,  5],   ← 按同样顺序重新分行
   [6,  7,  8,  9, 10, 11]]
```

### 6.6 `torch.arange()` — 等差序列

```python
torch.arange(12)          # tensor([0, 1, 2, ..., 11])
torch.arange(2, 10)       # tensor([2, 3, 4, ..., 9])
torch.arange(0, 1, 0.2)   # tensor([0.0, 0.2, 0.4, 0.6, 0.8])
```

| 参数 | 含义 |
|------|------|
| `arange(end)` | 从 0 到 end-1 |
| `arange(start, end)` | 从 start 到 end-1 |
| `arange(start, end, step)` | 步长为 step |

注意：**不包含 end**（左闭右开）。

### 6.7 广播机制（Broadcasting）

当两个张量形状不同时，PyTorch 自动"扩展"较小的张量以匹配较大的：

```python
a = torch.arange(3).reshape(3, 1)   # 形状 (3, 1)
b = torch.arange(2).reshape(1, 2)   # 形状 (1, 2)
c = a + b                           # 形状 (3, 2)
```

**广播规则（从最右维度开始逐维比较）：**

```
步骤 1：对齐维度（右对齐）
  a:  3 × 1
  b:  1 × 2

步骤 2：每个维度检查——两个维度兼容当且仅当：
  - 相等，或
  - 其中一个为 1

步骤 3：大小为 1 的维度被"复制扩展"到另一个的大小
  a 扩展:  3 × 2  （列方向复制）
  b 扩展:  3 × 2  （行方向复制）

步骤 4：逐元素运算
```

**具体过程：**

```
a = [[0],      b = [[0, 1]]
     [1],
     [2]]

a 扩展 → [[0, 0],     b 扩展 → [[0, 1],
          [1, 1],                [0, 1],
          [2, 2]]                [0, 1]]

a + b = [[0, 1],
         [1, 2],
         [2, 3]]
```

**常见广播场景：**

| a 形状 | b 形状 | 结果形状 | 说明 |
|--------|--------|----------|------|
| `(3, 4)` | `(4,)` | `(3, 4)` | 每行加一个向量 |
| `(3, 1)` | `(1, 4)` | `(3, 4)` | 外积式组合 |
| `(3, 4)` | `(1,)` | `(3, 4)` | 加一个标量 |
| `(2, 3, 4)` | `(3, 4)` | `(2, 3, 4)` | 低维自动升维 |

### 6.8 索引与切片

```python
B = torch.arange(12).reshape(3, 4).float()
# tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11.]])
```

**读取：**

```python
B[0]         # 第 0 行 → tensor([0., 1., 2., 3.])
B[0, 1]      # 第 0 行第 1 列 → tensor(1.)
B[:2]        # 前 2 行 → 形状 (2, 4)
B[:, 1]      # 所有行的第 1 列 → 形状 (3,)
B[1:3, 2:4]  # 第 1~2 行、第 2~3 列 → 形状 (2, 2)
```

**赋值（原地修改）：**

```python
B[:2, :] = 9    # 前两行全部设为 9
B[0, 0] = 99    # 单个元素赋值
B[:, -1] = 0    # 最后一列全设为 0
```

**索引语法 `[行, 列]` 备忘：**

| 写法 | 含义 |
|------|------|
| `B[i]` | 第 i 行（整行） |
| `B[i, j]` | 第 i 行第 j 列（单元素） |
| `B[i:j]` | 第 i 到 j-1 行 |
| `B[:, j]` | 所有行的第 j 列 |
| `B[:2, :]` | 前 2 行，所有列 |
| `B[-1]` | 最后一行 |

### 6.9 `.float()` — 类型转换

```python
B = torch.arange(12).reshape(3, 4).float()
```

- `torch.arange(12)` 默认生成 `int64`
- `.float()` 转换为 `float32`
- 深度学习中几乎所有数据都需要是 `float32`

其他转换方法：

```python
x.int()      # → int32
x.long()     # → int64
x.double()   # → float64
x.bool()     # → bool
x.to(torch.float16)  # → float16（半精度）
```

### 6.10 `torch.cat()` — 拼接张量

```python
X = torch.tensor([[1, 2], [3, 4]])   # 形状 (2, 2)
Y = torch.tensor([[1, 0], [3, 5]])   # 形状 (2, 2)

Z_row = torch.cat((X, Y), dim=0)     # 按行拼接 → (4, 2)
Z_col = torch.cat((X, Y), dim=1)     # 按列拼接 → (2, 4)
```

| 参数 | 含义 |
|------|------|
| 第一个参数 | 张量的**元组或列表** `(X, Y)` |
| `dim=0` | 沿第 0 维（行方向）拼接，行数增加 |
| `dim=1` | 沿第 1 维（列方向）拼接，列数增加 |

**图解：**

```
dim=0（上下堆叠）：          dim=1（左右拼接）：
  X: [[1, 2],                X|Y: [[1, 2, 1, 0],
      [3, 4],                      [3, 4, 3, 5]]
  Y:  [1, 0],                形状: (2, 4)
      [3, 5]]
  形状: (4, 2)
```

**拼接要求：** 除了 `dim` 指定的维度外，其他维度的大小必须相同。

### 6.11 逻辑比较 `==`

```python
mask = X == Y
# tensor([[ True, False],
#         [ True, False]])
```

- 逐元素比较，返回 **bool 类型张量**
- 形状与输入相同
- `True` 表示对应位置相等，`False` 表示不等

其他比较运算符：

| 运算符 | 含义 |
|--------|------|
| `==` | 等于 |
| `!=` | 不等于 |
| `>` | 大于 |
| `<` | 小于 |
| `>=` | 大于等于 |
| `<=` | 小于等于 |

### 6.12 原地操作（In-place Operations）

```python
# 非原地（创建新对象）
X = X + Y          # 新建一个张量，X 指向新对象，id(X) 改变

# 原地（修改原对象）
X += Y             # 等价于 X.add_(Y)，不创建新张量，id(X) 不变
X[:] = X + Y       # 也是原地：把结果写回 X 的内存
```

| 方式 | 是否创建新对象 | id 是否改变 |
|------|----------------|-------------|
| `X = X + Y` | ✅ 是 | ✅ 改变 |
| `X += Y` | ❌ 否 | ❌ 不变 |
| `X.add_(Y)` | ❌ 否 | ❌ 不变 |
| `X[:] = X + Y` | ❌ 否 | ❌ 不变 |

**为什么原地操作重要？**
- 深度学习中张量很大（百万级参数），频繁创建新对象浪费内存
- 带下划线 `_` 后缀的方法都是原地版本：`add_()`, `mul_()`, `zero_()`, `fill_()`

### 6.13 `.item()` — 张量 → Python 标量

```python
t = torch.tensor([3.5])
scalar = t.item()       # 3.5，类型是 Python float
```

- **只能用于单元素张量**（`numel() == 1`）
- 返回 Python 原生数值类型（`int` / `float`）
- 用途：打印 loss 值、做条件判断等需要 Python 数值的场景

```python
# 错误用法
torch.tensor([1, 2, 3]).item()  # RuntimeError! 不是单元素
```

### 6.14 `.numpy()` — Tensor ↔ NumPy 转换

```python
# Tensor → NumPy
arr = t.numpy()                    # 共享内存！修改一个另一个也变

# NumPy → Tensor
t2 = torch.tensor(arr)            # 复制数据，不共享
t3 = torch.from_numpy(arr)        # 共享内存
```

| 方法 | 方向 | 是否共享内存 |
|------|------|-------------|
| `tensor.numpy()` | Tensor → NumPy | ✅ 共享（CPU 上） |
| `torch.tensor(ndarray)` | NumPy → Tensor | ❌ 复制 |
| `torch.from_numpy(ndarray)` | NumPy → Tensor | ✅ 共享 |

注意：GPU 上的张量不能直接 `.numpy()`，需先 `.cpu().numpy()`。

---

## 七、HW1 逐题解析

### 题 1：pandas 数据预处理流水线

**完整流程图：**

```
原始 CSV 字符串
    ↓  pd.read_csv(StringIO(...))
DataFrame（含 NaN）
    ↓  iloc[:, :2] / iloc[:, -1]
inputs（前两列）、outputs（最后一列）
    ↓  select_dtypes(include='number')
找出数值列（NumRooms）
    ↓  fillna(mean())
数值列 NaN → 均值 3.0
    ↓  get_dummies(dummy_na=True)
类别列（Alley）→ One-Hot 编码
    ↓  to_numpy(dtype=float) → torch.tensor()
转为 PyTorch 张量 X, y
```

**数据变化过程：**

```
原始:                           填充后:                         One-Hot后:
NumRooms  Alley   Price        NumRooms  Alley                 NumRooms  Alley_Pave  Alley_nan
NaN       Pave    127500       3.0       Pave                  3.0       1.0         0.0
2.0       NaN     106000  →    2.0       NaN              →    2.0       0.0         1.0
4.0       NaN     178100       4.0       NaN                   4.0       0.0         1.0
NaN       NaN     140000       3.0       NaN                   3.0       0.0         1.0
```

最终：`X.shape = (4, 3)`，`y.shape = (4,)`

### 题 2：reshape

```python
A = M.reshape(2, 6)
```

把 (3,4)=12 个元素重排为 (2,6)=12 个元素。元素按行优先顺序不变。

### 题 3：广播

```python
a = torch.arange(3).reshape(3, 1)   # [[0],[1],[2]]
b = torch.arange(2).reshape(1, 2)   # [[0, 1]]
```

(3,1) + (1,2) → 广播为 (3,2) + (3,2) → 结果 (3,2)。

### 题 4：切片赋值

```python
B[:2, :] = 9
```

`:2` = 行索引 0 和 1，`:` = 所有列。将前两整行设为 9。

### 题 5：cat 拼接

```python
torch.cat((X, Y), dim=0)   # 行方向堆叠 (2,2)+(2,2) → (4,2)
torch.cat((X, Y), dim=1)   # 列方向拼接 (2,2)+(2,2) → (2,4)
```

### 题 6：原地操作 + 标量转换

```python
X += Y          # 原地加法，id 不变
t.item()        # 单元素张量 → Python float
t.numpy()       # 张量 → numpy 数组
```

---

## 附录：常见报错与解决

| 报错信息 | 原因 | 解决 |
|----------|------|------|
| `NameError: name 'pd' is not defined` | 没有运行 `import pandas as pd` 的 cell | 先运行导入 cell（从头 Run All） |
| `NameError: name 'torch' is not defined` | 没有运行 `import torch` 的 cell | 同上 |
| `RuntimeError: shape '[2, 5]' is invalid for input of size 12` | reshape 前后元素数不匹配 | 确保维度乘积相等 |
| `RuntimeError: a]Tensor with 3 elements cannot be converted to Scalar` | 对多元素张量调用 `.item()` | 只对单元素张量用 `.item()` |
| `SettingWithCopyWarning` | 修改的是 DataFrame 的视图而非副本 | 用 `.copy()` 创建独立副本 |
