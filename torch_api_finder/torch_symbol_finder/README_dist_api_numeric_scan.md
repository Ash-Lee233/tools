# Torch Distributed API Numeric Scan

## 📌 简介

`dist_api_numeric_scan.py` 是一个静态分析工具，用于扫描一组 `torch.distributed` API，并判断这些接口：

> ✅ 是否涉及**数值计算**（例如 Reduce / SUM / MAX 等规约操作）  
> ❌ 还是仅做通信、同步或调度（broadcast / gather / barrier 等）

输出格式为：

```
API:涉及
API:不涉及
```

适合用于：

- 分布式 API 分类
- 自动化兼容性分析
- 后端迁移（CUDA / NPU / XPU）前的能力审计
- 测试用例生成前的能力判断

---

## 📂 文件说明

```
dist_api_numeric_scan.py   # 主脚本
apis.txt                  # （可选）API 列表文件
```

---

## 🚀 使用方法

### ✅ 方式一：命令行直接输入 API

```bash
python dist_api_numeric_scan.py \
  torch.distributed.all_reduce \
  torch.distributed.broadcast \
  torch.distributed.barrier
```

---

### ✅ 方式二：从文件读取 API

准备 `apis.txt`（每行一个接口，支持注释）：

```txt
# reduce ops
torch.distributed.all_reduce
torch.distributed.reduce_scatter

# comm only
torch.distributed.broadcast
torch.distributed.all_gather
torch.distributed.barrier
```

运行：

```bash
python dist_api_numeric_scan.py --apis-file apis.txt
```

---

## 📤 输出格式

标准输出为：

```
torch.distributed.all_reduce:涉及
torch.distributed.broadcast:不涉及
torch.distributed.barrier:不涉及
```

---

## 🔍 判定依据（核心逻辑）

脚本采用 **多层启发式规则**：

---

### 🥇 第一优先级：源码扫描（若可 import）

如果 API 能从当前 Python 环境 import：

- 使用 `inspect` 定位其源码文件
- 扫描是否包含以下特征：

#### ✔ 数值计算特征（命中即判为“涉及”）

- `ReduceOp`
- `SUM / MAX / MIN / PRODUCT / AVG`
- `allreduce`
- `reduce_scatter`
- `reduce`
- `op=`

#### ✖ 纯通信特征

- `broadcast`
- `all_gather`
- `all_to_all`
- `send / recv`
- `barrier`
- `init_process_group`

---

### 🥈 第二优先级：API 名称规则

若无法定位源码（例如 C++ binding），则根据名称推断：

#### 判为【涉及】

- `all_reduce`
- `reduce`
- `reduce_scatter`
- `scan`

#### 判为【不涉及】

- `broadcast`
- `gather`
- `scatter`
- `all_to_all`
- `send / recv`
- `barrier`
- `init_process_group`

---

### 🥉 最终兜底策略

如果源码和名字都无法判断：

➡ 默认判定：

```
不涉及
```

（保守策略，避免误把纯通信接口当作数值计算）

---

## ⚠️ 局限说明

本工具属于：

> **静态启发式分析，不执行代码**

可能存在：

- 深层 C++ backend 调用未被扫描
- wrapper 函数掩盖 reduce 行为
- 特殊定制后端（Ascend / XPU）差异

如果用于关键决策（例如算子迁移），建议：

✔ 结合运行时测试  
✔ 结合 backend 实现源码  
✔ 增加 repo 扫描模式  

---

## 💡 典型判定原则总结

| 接口类型 | 示例 | 是否涉及数值计算 |
|---------|------|---------------|
| Reduce / 规约 | all_reduce | ✅ |
| ReduceScatter | reduce_scatter | ✅ |
| Gather / 拼接 | all_gather | ❌ |
| Broadcast | broadcast | ❌ |
| 同步 | barrier | ❌ |
| 建联 | init_process_group | ❌ |

---

如果你愿意，我也可以：

✔ 升级脚本支持 **torch 源码仓扫描模式**  
✔ 输出 CSV / Excel  
✔ 输出三分类（规约计算 / 数据重排 / 控制）  
✔ 生成测试模板  

直接说你想往哪一步工程化演进 😄
