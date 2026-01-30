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

# Export Excel report
python -m hccl_analyzer --repo-path /path/to/repo --excel report.xlsx
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
                     [--json] [--output OUTPUT] [--excel FILE]

options:
  --repo-path REPO_PATH   Path to a local Ascend PyTorch repository
  --repo-url REPO_URL     Git URL to shallow-clone before analysis
  --json                  Emit structured JSON instead of text
  --output, -o OUTPUT     Write output to file instead of stdout
  --excel FILE            Export results to an Excel (.xlsx) file
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

**Understanding HCCL References**: The "HCCL references" section shows specific code locations where the API directly uses HCCL components (classes, imports, operators). These references prove the API requires HCCL runtime support—without HCCL, the code would fail to import, instantiate, or execute HCCL-specific functionality.

### Excel (`--excel`)

Generates an `.xlsx` file with color-coded rows by tier:

| Tier | API | File | References |
|------|-----|------|------------|
| Tier 1 (HIGH) | `torch.distributed.gather` | `torch_npu\__init__.py:198` | `L247 [ProcessGroupHCCL]: ...` |
| Tier 2 (MEDIUM) | ... | ... | `L134: _group.allgather(...)` |
| Tier 3 (LOW) | ... | ... | `func_a -> func_b -> func_c` |

- Tier 1 rows: blue background — shows key HCCL patterns (`ProcessGroupHCCL`, `"hccl"`, etc.), filtered and deduplicated, max 5 entries
- Tier 2 rows: yellow background — shows ProcessGroup dispatch calls
- Tier 3 rows: orange background — shows the inferred call chain
- File paths are relative to the repository root

Requires `openpyxl` (`pip install openpyxl`).

#### Understanding References — Why These APIs Need HCCL

The **References** column provides concrete evidence showing why each API depends on HCCL capabilities:

**Tier 1 (HIGH confidence) References:**
- **Direct HCCL usage**: Shows specific code locations where the API directly imports, instantiates, or calls HCCL-related components (e.g., `ProcessGroupHCCL`, `"hccl"` backend name, `hcom` operators)
- **Why it matters**: These references prove the API **cannot function without HCCL** because it explicitly uses HCCL classes, checks HCCL availability, or imports NPU-specific distributed modules
- **Example**: `L247 [ProcessGroupHCCL]: pg = ProcessGroupHCCL(...)` means the API creates an HCCL process group, requiring HCCL runtime support

**Tier 2 (MEDIUM confidence) References:**
- **ProcessGroup dispatch**: Shows calls to collective operations (`allreduce`, `broadcast`, etc.) through the ProcessGroup abstraction
- **Why it matters**: On NPU devices, these ProcessGroup calls **automatically route to ProcessGroupHCCL** at runtime. The API relies on HCCL's implementation of collective communication primitives
- **Example**: `L134: _group.allgather(...)` means the API calls `allgather`, which on NPU devices executes via HCCL's `hcom_allgather` under the hood

**Tier 3 (LOW confidence) References:**
- **Call chain**: Shows the inferred function call path from the API to HCCL-related code
- **Why it matters**: The API indirectly depends on HCCL through intermediate functions. While not direct, the dependency chain demonstrates that HCCL capabilities are required somewhere in the execution path
- **Example**: `func_a -> func_b -> _hccl_allreduce` shows the API eventually reaches HCCL code through a call chain

**Key insight**: References are not just code locations—they are **proof points** that demonstrate the API's dependency on HCCL. Without HCCL runtime support, these APIs would fail or behave incorrectly on NPU devices.

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
        "references": [
          {
            "file": ".../distributed_c10d.py",
            "line": 127,
            "pattern": "HCCL",
            "context": "warnings.warn(\"HCCL doesn't support gather ...\")"
          }
        ],
        "chain": [...]
      }
    ],
    "medium": [...],
    "low": [...]
  }
}
```

**Understanding the `references` field**: Each entry in `references` contains:
- `file` & `line`: Exact code location where HCCL dependency is detected
- `pattern`: The HCCL-related pattern matched (e.g., `"ProcessGroupHCCL"`, `"hccl"`, `"hcom"`)
- `context`: The actual source code line showing how HCCL is used

These references serve as **evidence** that the API requires HCCL capabilities—they show where and how the code depends on HCCL components.

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
- No third-party dependencies for text/JSON output (stdlib only)
- `openpyxl` for Excel export (`pip install openpyxl`)
