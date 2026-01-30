"""File scanning and pattern matching for HCCL analysis."""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Iterator

from .models import DistributedAPI, HCCLReference

# ---------------------------------------------------------------------------
# HCCL detection patterns
# ---------------------------------------------------------------------------

HCCL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bProcessGroupHCCL\b"),
    re.compile(r"""['"]hccl['"]"""),
    re.compile(r"\bimport\s+.*hccl\b", re.IGNORECASE),
    re.compile(r"\bfrom\s+.*hccl\b", re.IGNORECASE),
    re.compile(r"\btorch_npu\.distributed\b"),
    re.compile(r"\btorch_npu\._C\._distributed_c10d\b"),
    re.compile(r"\bProcessGroupNPU\b"),
    re.compile(r"\bhcom\b"),
    re.compile(r"\bHCCL\b"),
    re.compile(r"\bhccl_available\b"),
]

# Directories to skip during traversal
_SKIP_DIRS = {
    ".git", "__pycache__", ".tox", ".mypy_cache",
    "node_modules", ".eggs", "build", "dist",
    "third_party", ".github",
}


def iter_python_files(root: str | Path) -> Iterator[Path]:
    """Yield all ``*.py`` files under *root*, skipping uninteresting dirs."""
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        # prune in-place
        dirnames[:] = [
            d for d in dirnames
            if d not in _SKIP_DIRS and not d.endswith(".egg-info")
        ]
        for fn in filenames:
            if fn.endswith(".py"):
                yield Path(dirpath) / fn


# ---------------------------------------------------------------------------
# HCCL reference scanning
# ---------------------------------------------------------------------------

def scan_hccl_references(filepath: Path) -> list[HCCLReference]:
    """Return all HCCL pattern matches found in *filepath*."""
    refs: list[HCCLReference] = []
    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return refs

    for lineno, line in enumerate(text.splitlines(), start=1):
        for pat in HCCL_PATTERNS:
            if pat.search(line):
                refs.append(HCCLReference(
                    file=str(filepath),
                    line=lineno,
                    pattern=pat.pattern,
                    context=line,
                ))
                break  # one match per line is enough
    return refs


# ---------------------------------------------------------------------------
# Public API surface extraction
# ---------------------------------------------------------------------------

# Paths (relative to repo root) that may define distributed public APIs.
# Covers both standard PyTorch layout AND torch_npu plugin layout.
_DISTRIBUTED_PATH_FRAGMENTS = (
    os.path.join("torch", "distributed"),
    os.path.join("torch_npu", "distributed"),
    os.path.join("torch_npu", "csrc", "distributed"),
)


def _is_distributed_path(filepath: Path, repo_root: Path) -> bool:
    try:
        rel = str(filepath.relative_to(repo_root))
    except ValueError:
        return False
    # Normalise separators for Windows
    rel = rel.replace("\\", "/")
    return any(
        frag.replace("\\", "/") in rel
        for frag in _DISTRIBUTED_PATH_FRAGMENTS
    )


def _qualified_module(filepath: Path, repo_root: Path) -> str:
    """Convert a file path to a dotted module name."""
    try:
        rel = filepath.relative_to(repo_root)
    except ValueError:
        return str(filepath)
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def extract_public_apis(filepath: Path, repo_root: Path) -> list[DistributedAPI]:
    """Extract public functions / classes from a distributed module file."""
    if not _is_distributed_path(filepath, repo_root):
        return []

    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(filepath))
    except (OSError, SyntaxError):
        return []

    module = _qualified_module(filepath, repo_root)
    apis: list[DistributedAPI] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            apis.append(DistributedAPI(
                qualified_name=f"{module}.{node.name}",
                file=str(filepath),
                line=node.lineno,
                kind="function",
            ))
        elif isinstance(node, ast.AsyncFunctionDef) and not node.name.startswith("_"):
            apis.append(DistributedAPI(
                qualified_name=f"{module}.{node.name}",
                file=str(filepath),
                line=node.lineno,
                kind="function",
            ))
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            apis.append(DistributedAPI(
                qualified_name=f"{module}.{node.name}",
                file=str(filepath),
                line=node.lineno,
                kind="class",
            ))

    return apis


# ---------------------------------------------------------------------------
# Monkey-patch detection
# ---------------------------------------------------------------------------

# Pattern: ``torch.distributed.XXX = ...``
_MONKEY_PATCH_RE = re.compile(
    r"^\s*(torch\.distributed(?:\.\w+)*\.(\w+))\s*=\s*(.+)",
)


def extract_monkey_patches(filepath: Path) -> list[DistributedAPI]:
    """Find ``torch.distributed.X = ...`` assignments that patch the public API."""
    apis: list[DistributedAPI] = []
    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return apis

    for lineno, line in enumerate(text.splitlines(), start=1):
        m = _MONKEY_PATCH_RE.search(line)
        if m:
            full_target = m.group(1)
            apis.append(DistributedAPI(
                qualified_name=full_target,
                file=str(filepath),
                line=lineno,
                kind="monkey-patch",
            ))
    return apis


# ---------------------------------------------------------------------------
# Intra-file call graph helpers
# ---------------------------------------------------------------------------

def extract_function_calls(filepath: Path) -> dict[str, set[str]]:
    """Return a mapping *function_name -> {called_names}* for top-level defs.

    This is a lightweight, best-effort extraction used by the BFS in
    ``analyzer.py``.  It considers simple ``Name`` and ``Attribute`` call
    nodes.
    """
    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(filepath))
    except (OSError, SyntaxError):
        return {}

    graph: dict[str, set[str]] = {}

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        calls: set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func = child.func
                if isinstance(func, ast.Name):
                    calls.add(func.id)
                elif isinstance(func, ast.Attribute):
                    calls.add(func.attr)
        graph[node.name] = calls

    return graph
