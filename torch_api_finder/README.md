# torch-symbol-finder

一个小工具，用来在你**当前** Python 环境中（例如 `torch==2.7.1`）自动探测：
你手头的一批 `torch.xxx.yyy` **dotted symbol** 是否真的可用，以及它对应的“真实可 import 路径”。

> 适合排查：某些内部/私有符号（例如 `torch._C.*`, `torch._dynamo.*`）在不同 build（CPU-only / CUDA / ROCm / vendor）中
> 是否存在、是否导出、路径是否变化。

## 安装/使用（源码方式）

解压 zip 后，在项目根目录执行：

```bash
python -m torch_symbol_finder.cli
```

也可以从文件读取你自己的 symbol 列表：

```bash
python -m torch_symbol_finder.cli --symbols-file symbols.txt
```

输出 JSON（便于后续自动处理）：

```bash
python -m torch_symbol_finder.cli --json > report.json
```

输出 Excel 文件（便于分享和查看）：

```bash
python -m torch_symbol_finder.cli --excel report.xlsx
```

> Excel 输出依赖 `openpyxl`，请先安装：`pip install openpyxl`

如果你希望更快一些（不扫描 `torch.*` 子模块的 `__all__` 作为 hint）：

```bash
python -m torch_symbol_finder.cli --no-hints
```

## symbols.txt 格式

- 每行一个 dotted symbol
- 支持 `#` 注释
- 也支持一行里用逗号分隔多个 symbol

示例：

```text
# cuda internals (可能要求 CUDA build)
torch._C._cuda_getDeviceCount
torch.cuda.device_count

# typo example
torch.zeroes
```

## 输出说明

- `[OK]` 表示：在你当前环境中确实能解析到对象
  - 会给出推荐的 import 形态：`from <module> import <name>`（以及后续属性链）
- `[MISS]` 表示：当前环境中没有该符号（或未导出）
  - 如果开启 hints，会给出一些"可能导出同名符号"的模块提示（best-effort）

使用 `--excel` 时，输出为格式化的 `.xlsx` 文件，包含四列：

| 列名 | 说明 |
|------|------|
| 原始接口 | 原始 dotted symbol |
| 正确import格式 | 推荐的 `from ... import ...` 写法 |
| 备注 | 额外属性链或 hint 模块列表 |
| 状态 | `OK`（绿色）/ `MISS`（红色）/ `HINT`（黄色）|

## 注意

- 很多 `torch._C._cuda_*` 需要 **CUDA 版** PyTorch 才会存在。CPU-only wheel 通常会缺失。
- `torch` 的内部符号在小版本/不同发行版间变化很常见。这个工具的目标是“在你本机的真实环境里查清楚”。

## 许可证

MIT
