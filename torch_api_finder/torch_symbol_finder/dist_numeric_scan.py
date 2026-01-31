#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
给定一组 torch.distributed 相关 API（dotted path），静态分析其是否涉及“数值计算”。
输出格式：API:涉及 / API:不涉及

使用方式：
  1) 直接在命令行传入：
     python dist_api_numeric_scan.py torch.distributed.all_reduce torch.distributed.broadcast

  2) 从 txt 文件读取（每行一个 API，支持 # 注释）：
     python dist_api_numeric_scan.py --apis-file apis.txt
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


# -----------------------------
# 规则库：名称启发式（兜底）
# -----------------------------
# 明确“涉及数值计算”的关键词（reduce/scan/accumulate类）
NAME_POSITIVE = (
    "all_reduce",
    "reduce_scatter",
    "reduce",
    "allreduce",          # C++/backend常见命名
    "reducescatter",
    "scan",
    "inclusive_scan",
    "exclusive_scan",
)

# 明确“不涉及数值计算”的关键词（纯通信/重排/控制类）
NAME_NEGATIVE = (
    "all_gather",
    "allgather",
    "gather",
    "scatter",
    "broadcast",
    "all_to_all",
    "alltoall",
    "send",
    "recv",
    "isend",
    "irecv",
    "barrier",
    "init_process_group",
    "destroy_process_group",
    "new_group",
    "rendezvous",
    "get_rank",
    "get_world_size",
    "wait",
    "poll",
)

# -----------------------------
# 规则库：源码特征（更强证据）
# -----------------------------
# 数值计算（规约）相关特征：出现这些通常说明“涉及”
SOURCE_POSITIVE_PATTERNS = [
    r"\bReduceOp\b",
    r"\bReductionOp\b",
    r"\bop\s*=",
    r"\bSUM\b|\bMAX\b|\bMIN\b|\bPRODUCT\b|\bAVG\b",
    r"\ball_reduce\b|\ballreduce\b",
    r"\breduce_scatter\b|\breducescatter\b",
    r"\breduce\b",
]

# 明确纯通信/控制特征：只出现这些（且不出现 positive）通常“不涉及”
SOURCE_NEGATIVE_PATTERNS = [
    r"\bbroadcast\b",
    r"\ball_gather\b|\ballgather\b",
    r"\bgather\b",
    r"\bscatter\b",
    r"\ball_to_all\b|\balltoall\b",
    r"\bsend\b|\brecv\b|\bisend\b|\birecv\b",
    r"\bbarrier\b",
    r"\binit_process_group\b|\bdestroy_process_group\b|\bnew_group\b",
]


@dataclass
class ResolveInfo:
    ok: bool
    obj: Optional[object] = None
    src_file: Optional[str] = None
    err: Optional[str] = None


def read_apis_from_file(path: Path) -> List[str]:
    apis: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        apis.append(s)
    return apis


def resolve_dotted(name: str) -> ResolveInfo:
    """
    尝试解析 dotted path 到 Python 对象，并拿到其源码文件路径（如果有）。
    """
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        modname = ".".join(parts[:i])
        attrpath = parts[i:]
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue

        obj = mod
        ok = True
        for a in attrpath:
            if not hasattr(obj, a):
                ok = False
                break
            obj = getattr(obj, a)
        if not ok:
            continue

        src_file = None
        try:
            src_file = inspect.getsourcefile(obj) or inspect.getfile(obj)
        except Exception:
            # built-in / C++ bound object 可能拿不到
            src_file = None

        return ResolveInfo(ok=True, obj=obj, src_file=src_file)

    return ResolveInfo(ok=False, err="cannot import/resolve")


def extract_source_snippet(obj: object, src_file: Optional[str], radius_lines: int = 220) -> str:
    """
    提取用于扫描的源码片段：
    - 优先用 inspect.getsource(obj)
    - 否则读取 src_file 全文（或部分）
    """
    # 1) 尝试直接拿函数/类源码（最精确）
    try:
        src = inspect.getsource(obj)
        if src and len(src) > 0:
            return src
    except Exception:
        pass

    # 2) 退化：读取文件（可能很大，适当截断）
    if src_file and os.path.exists(src_file) and os.path.isfile(src_file):
        try:
            text = Path(src_file).read_text(encoding="utf-8", errors="ignore")
            # 为了速度，截断前 2000 行
            lines = text.splitlines()
            if len(lines) > 2000:
                lines = lines[:2000]
            return "\n".join(lines)
        except Exception:
            return ""

    return ""


def name_heuristic(api: str) -> Optional[bool]:
    """
    根据名字快速判定：
    True=涉及；False=不涉及；None=无法判定
    """
    low = api.lower()

    for k in NAME_POSITIVE:
        if k in low:
            return True
    for k in NAME_NEGATIVE:
        if k in low:
            return False
    return None


def source_heuristic(source: str) -> Optional[bool]:
    """
    根据源码片段特征判定：
    True=涉及；False=不涉及；None=无法判定
    """
    if not source:
        return None

    pos = any(re.search(p, source) for p in SOURCE_POSITIVE_PATTERNS)
    neg = any(re.search(p, source) for p in SOURCE_NEGATIVE_PATTERNS)

    # 强规则：
    # - 只要命中 reduce 类特征 → 判“涉及”
    if pos:
        return True

    # - 没命中 pos，但命中大量纯通信特征 → 判“不涉及”
    if neg and not pos:
        return False

    return None


def classify(api: str) -> bool:
    """
    最终分类：True=涉及；False=不涉及
    策略：
      1) 若能 import -> 源码特征优先
      2) 否则/特征不足 -> 名称规则
      3) 仍不确定 -> 默认“不涉及”（保守：避免把纯通信误判成计算）
    """
    info = resolve_dotted(api)
    if info.ok:
        snippet = extract_source_snippet(info.obj, info.src_file)
        s = source_heuristic(snippet)
        if s is not None:
            return s

    n = name_heuristic(api)
    if n is not None:
        return n

    return False


def main() -> int:
    p = argparse.ArgumentParser(description="Scan torch.distributed APIs and classify whether they involve numerical compute.")
    p.add_argument("--apis-file", type=str, default=None, help="txt file path, one API per line (supports # comments).")
    p.add_argument("apis", nargs="*", help="APIs as dotted paths, e.g. torch.distributed.all_reduce")
    args = p.parse_args()

    apis: List[str] = []
    if args.apis_file:
        apis.extend(read_apis_from_file(Path(args.apis_file)))
    apis.extend(args.apis)

    if not apis:
        print("No APIs provided.")
        return 2

    for api in apis:
        involves = classify(api)
        print(f"{api}:{'涉及' if involves else '不涉及'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
