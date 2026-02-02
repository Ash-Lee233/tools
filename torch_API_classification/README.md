# PyTorch API 算子分类工具

枚举本地 PyTorch 的所有 ATen 算子，按实现方式分为 4 类，结果保存为 Excel。

## 环境要求

- Python 3.8+
- PyTorch（已在 2.10.0 上测试）
- openpyxl（脚本会自动安装）

## 使用方法

```bash
python classify_torch_ops.py
```

运行后会在当前目录生成 `pytorch_api_classification.xlsx`。

## 分类定义

| 编号 | 分类名称 | 含义 | 检测方式 |
|------|----------|------|----------|
| 1 | CUDA直接提供算子 | 调用 cuBLAS/cuDNN/cuFFT/cuSOLVER 等 CUDA 库 | 白名单匹配 |
| 2 | CUDA C++自定义算子 | 有专门 CUDA C++ kernel（非库调用） | per-backend kernel 且不在库映射白名单中 |
| 3 | 算子拼接 | 通过组合其他算子实现的复合算子 | CompositeImplicitAutograd / CompositeExplicitAutograd |
| 4 | Triton开发 | 已有 Triton 实现的算子 | 在 `torch._inductor.lowering.lowerings` 中且不在 fallbacks 中 |

### 分类优先级

一个算子可能同时有多种实现，按以下优先级判定主分类：

1. **算子拼接（Composite）** — CompositeImplicitAutograd/Explicit 优先
2. **Triton lowering** — 有 inductor lowering 且不是 CUDA 库算子
3. **CUDA 库算子** — 匹配 cuBLAS/cuDNN/cuFFT/cuSOLVER 白名单
4. **CUDA C++ 自定义算子** — 其余有 per-backend kernel 的算子

## 输出 Excel 说明

文件：`pytorch_api_classification.xlsx`

| 列名 | 说明 | 示例 |
|------|------|------|
| 算子名称 | ATen 算子全名 | `aten::mm`, `aten::add.Tensor` |
| 公共API名称 | 用户常用的公共 API 名称 | `torch.mm`, `torch.add` |
| 分类编号 | 1 / 2 / 3 / 4 | 1 |
| 分类名称 | 对应的中文分类名称 | CUDA直接提供算子 |
| 是否有Triton lowering | 无论主分类如何，标注该算子是否有 Triton 实现 | 是 / 否 |
| Tags | 算子标签 | core, pointwise, pt2_compliant_tag |

## 最近一次运行结果（PyTorch 2.10.0）

共计 **3835** 个 ATen 算子，分布如下：

| 分类 | 数量 |
|------|------|
| 1-CUDA直接提供算子 | 105 |
| 2-CUDA C++自定义算子 | 1439 |
| 3-算子拼接 | 1665 |
| 4-Triton开发 | 626 |

典型算子分类示例：

| 算子 | 分类 | 说明 |
|------|------|------|
| `aten::mm` | 1-CUDA直接提供算子 | 矩阵乘法，调用 cuBLAS |
| `aten::conv2d` | 3-算子拼接 | 复合算子，分解为底层卷积操作 |
| `aten::relu` | 4-Triton开发 | 有 Triton lowering 实现 |
| `aten::add.Tensor` | 4-Triton开发 | 有 Triton lowering 实现 |
| `aten::fft_fft` | 3-算子拼接 | 复合算子，分解为 `_fft_r2c`/`_fft_c2c` |
| `aten::linalg_svd` | 3-算子拼接 | 复合算子，分解为 `_linalg_svd` |
