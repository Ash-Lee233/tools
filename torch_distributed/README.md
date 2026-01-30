# HCCL Dependency Analyzer

Static analysis tool that scans Ascend PyTorch (`torch_npu`) source code to identify which `torch.distributed` APIs depend on HCCL (Huawei Collective Communication Library) interfaces.

## Quick Start

```bash
cd torch_distributed

# Analyze a local repository
python -m hccl_analyzer --repo-path /path/to/ascend-pytorch

# Auto-clone and analyze
python -m hccl_analyzer --repo-url https://gitcode.com/Ascend/pytorch.git

# JSON output
python -m hccl_analyzer --repo-path /path/to/repo --json

# Write to file
python -m hccl_analyzer --repo-path /path/to/repo --output report.txt
```

## Install (optional)

```bash
pip install -e .
hccl-analyzer --repo-path /path/to/repo
```

## Analysis Algorithm

The analyzer uses a three-tier confidence model:

### Tier 1 — HIGH confidence

The API's source file **directly contains** HCCL-related patterns:

| Pattern | Example |
|---------|---------|
| `ProcessGroupHCCL` | C++ backend class reference |
| `"hccl"` | Backend name string literal |
| `torch_npu.distributed` | NPU distributed module import |
| `torch_npu._C._distributed_c10d` | NPU C++ binding import |
| `HCCL` | Uppercase HCCL references |
| `hcom` | HCCL communication operator |
| `ProcessGroupNPU` | NPU process group |
| `hccl_available` | HCCL availability check |

### Tier 2 — MEDIUM confidence

The API's source file **dispatches through ProcessGroup abstraction**, which routes to `ProcessGroupHCCL` at runtime on NPU devices:

- `pg.allreduce(...)`, `group.broadcast(...)`, etc.
- `_get_backend(torch.device("npu"))` — explicitly fetches NPU backend

### Tier 3 — LOW confidence

The API reaches HCCL through a **multi-hop call chain** (BFS, max depth 3):

```
API function → intermediate call → ... → HCCL-related function
```

## What Gets Scanned

The tool discovers APIs from two sources:

1. **Module definitions** — public functions and classes under `torch_npu/distributed/` (and `torch/distributed/` if present)
2. **Monkey-patches** — assignments like `torch.distributed.gather = ...` that patch the standard PyTorch API at runtime

## CLI Reference

```
usage: hccl-analyzer [-h] (--repo-path REPO_PATH | --repo-url REPO_URL)
                     [--json] [--output OUTPUT]

options:
  --repo-path REPO_PATH   Path to a local Ascend PyTorch repository
  --repo-url REPO_URL     Git URL to shallow-clone before analysis
  --json                  Emit structured JSON instead of text
  --output, -o OUTPUT     Write output to file instead of stdout
```

## Output Format

### Text (default)

```
======================================================================
HCCL Dependency Analysis Report
======================================================================
Repository : /path/to/repo
Total APIs scanned : 66
HCCL-dependent APIs: 35

----------------------------------------------------------------------
Tier 1 - HIGH confidence (direct HCCL reference)  [31 APIs]
----------------------------------------------------------------------
  torch.distributed.gather
    defined at .../distributed_c10d.py:81  (monkey-patch)
    HCCL references:
      - .../distributed_c10d.py:127  pattern='HCCL'
        warnings.warn("HCCL doesn't support gather ...")
  ...
```

### JSON (`--json`)

```json
{
  "repo_path": "/path/to/repo",
  "total_apis_scanned": 66,
  "hccl_dependent_apis": 35,
  "by_confidence": {
    "high": [
      {
        "api": "torch.distributed.gather",
        "file": ".../distributed_c10d.py",
        "line": 81,
        "kind": "monkey-patch",
        "confidence": "high",
        "references": [...],
        "chain": [...]
      }
    ],
    "medium": [...],
    "low": [...]
  }
}
```

## Project Structure

```
torch_distributed/
    pyproject.toml              # Package metadata & entry point
    README.md
    .gitignore
    hccl_analyzer/
        __init__.py             # Public exports
        __main__.py             # python -m entry point
        models.py               # Dataclasses: HCCLReference, DistributedAPI, AnalysisResult, etc.
        scanner.py              # File traversal, regex matching, AST extraction, monkey-patch detection
        analyzer.py             # Three-tier analysis engine (HCCLAnalyzer)
        cli.py                  # argparse CLI
```

## Requirements

- Python >= 3.10
- No third-party dependencies (stdlib only)
