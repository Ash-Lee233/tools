"""Three-tier HCCL dependency analysis engine."""

from __future__ import annotations

import re
from collections import deque
from pathlib import Path
from typing import Sequence

from .models import (
    AnalysisReport,
    AnalysisResult,
    Confidence,
    DependencyEdge,
    DistributedAPI,
    HCCLReference,
)
from .scanner import (
    extract_function_calls,
    extract_monkey_patches,
    extract_public_apis,
    iter_python_files,
    scan_hccl_references,
)

# ---------------------------------------------------------------------------
# ProcessGroup dispatch patterns (Tier 2)
# ---------------------------------------------------------------------------

# Collective operations that route through ProcessGroup -> ProcessGroupHCCL
_COLLECTIVE_OPS = {
    "all_reduce", "allreduce", "all_gather", "allgather",
    "broadcast", "reduce", "reduce_scatter", "barrier",
    "all_to_all", "alltoall", "gather", "scatter",
    "send", "recv", "isend", "irecv",
    "all_gather_into_tensor", "reduce_scatter_tensor",
    "all_reduce_coalesced", "all_gather_coalesced",
    "reduce_scatter_coalesced",
    "batch_isend_irecv",
    "allgather_base_uneven", "reduce_scatter_tensor_uneven",
    "all_gather_into_tensor_uneven",
}

_PG_DISPATCH_RE = re.compile(
    r"\b(?:ProcessGroup|_world\.default_pg|default_pg|group|pg|_group)\s*\.\s*("
    + "|".join(_COLLECTIVE_OPS)
    + r")\b"
)

# NPU device dispatch: code that gets backend for NPU device
_NPU_BACKEND_RE = re.compile(
    r"""_get_backend\s*\(\s*torch\.device\s*\(\s*['"]npu['"]\s*\)\s*\)"""
)


def _has_pg_dispatch(source: str) -> list[tuple[int, str, str]]:
    """Return ``(lineno, callee, line_text)`` for PG dispatch calls."""
    hits: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(source.splitlines(), start=1):
        m = _PG_DISPATCH_RE.search(line)
        if m:
            hits.append((lineno, m.group(1), line))
        if _NPU_BACKEND_RE.search(line):
            hits.append((lineno, "_get_backend(npu)", line))
    return hits


# ---------------------------------------------------------------------------
# Core analyser
# ---------------------------------------------------------------------------

class HCCLAnalyzer:
    """Run three-tier analysis on a repository."""

    def __init__(self, repo_path: str | Path) -> None:
        self.repo_path = Path(repo_path)
        # caches populated during analysis
        self._hccl_by_file: dict[str, list[HCCLReference]] = {}
        self._apis: list[DistributedAPI] = []
        self._call_graphs: dict[str, dict[str, set[str]]] = {}
        self._file_sources: dict[str, str] = {}
        self._hccl_files: set[str] = set()
        # cross-file: function name -> files containing HCCL refs where it is defined
        self._hccl_func_names: set[str] = set()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> AnalysisReport:
        self._scan_repo()
        seen: set[str] = set()
        results: list[AnalysisResult] = []

        for api in self._apis:
            if api.qualified_name in seen:
                continue
            result = self._analyse_api(api)
            if result is not None:
                seen.add(api.qualified_name)
                results.append(result)

        # sort by confidence (high first), then name
        order = {Confidence.HIGH: 0, Confidence.MEDIUM: 1, Confidence.LOW: 2}
        results.sort(key=lambda r: (order[r.confidence], r.api.qualified_name))

        report = AnalysisReport(
            repo_path=str(self.repo_path),
            total_apis=len(self._apis),
            results=results,
        )
        return report

    # ------------------------------------------------------------------
    # Phase 0: repository scan
    # ------------------------------------------------------------------

    def _scan_repo(self) -> None:
        for filepath in iter_python_files(self.repo_path):
            fstr = str(filepath)

            # scan for HCCL references
            refs = scan_hccl_references(filepath)
            if refs:
                self._hccl_by_file[fstr] = refs
                self._hccl_files.add(fstr)

            # extract public APIs from distributed modules
            apis = extract_public_apis(filepath, self.repo_path)
            self._apis.extend(apis)

            # extract monkey-patches (torch.distributed.X = ...)
            patches = extract_monkey_patches(filepath)
            self._apis.extend(patches)

        # build set of function names defined in HCCL-referencing files
        for fstr in self._hccl_files:
            graph = self._get_call_graph(fstr)
            self._hccl_func_names.update(graph.keys())

    # ------------------------------------------------------------------
    # Per-API analysis
    # ------------------------------------------------------------------

    def _analyse_api(self, api: DistributedAPI) -> AnalysisResult | None:
        # Tier 1: same-file HCCL reference
        t1 = self._tier1(api)
        if t1 is not None:
            return t1

        # Tier 2: PG dispatch in same file
        t2 = self._tier2(api)
        if t2 is not None:
            return t2

        # Tier 3: BFS multi-hop
        t3 = self._tier3(api)
        if t3 is not None:
            return t3

        return None

    # ------------------------------------------------------------------
    # Tier 1 - direct HCCL reference in same file
    # ------------------------------------------------------------------

    def _tier1(self, api: DistributedAPI) -> AnalysisResult | None:
        refs = self._hccl_by_file.get(api.file)
        if not refs:
            return None
        return AnalysisResult(
            api=api,
            confidence=Confidence.HIGH,
            references=refs,
        )

    # ------------------------------------------------------------------
    # Tier 2 - ProcessGroup abstraction dispatch
    # ------------------------------------------------------------------

    def _tier2(self, api: DistributedAPI) -> AnalysisResult | None:
        source = self._read_source(api.file)
        if source is None:
            return None

        hits = _has_pg_dispatch(source)
        if not hits:
            return None

        refs = [
            HCCLReference(file=api.file, line=ln, pattern=f"PG.{callee}", context=ctx)
            for ln, callee, ctx in hits
        ]
        chain = [
            DependencyEdge(
                caller=api.qualified_name,
                callee=f"ProcessGroup.{callee}",
                file=api.file,
                line=ln,
            )
            for ln, callee, _ctx in hits
        ]
        return AnalysisResult(
            api=api,
            confidence=Confidence.MEDIUM,
            references=refs,
            chain=chain,
        )

    # ------------------------------------------------------------------
    # Tier 3 - BFS call-chain (max depth 3)
    # ------------------------------------------------------------------

    def _tier3(self, api: DistributedAPI, max_depth: int = 3) -> AnalysisResult | None:
        graph = self._get_call_graph(api.file)
        if not graph:
            return None

        # Determine the starting function name from the API
        func_name = api.qualified_name.rsplit(".", 1)[-1]
        # For monkey-patches, strip leading underscore of the impl function
        if func_name.startswith("_"):
            func_name_alt = func_name[1:]
        else:
            func_name_alt = "_" + func_name

        start = None
        for candidate in (func_name, func_name_alt):
            if candidate in graph:
                start = candidate
                break
        if start is None:
            return None

        # BFS over call graph
        queue: deque[tuple[str, list[str]]] = deque()
        queue.append((start, [start]))
        visited: set[str] = {start}

        while queue:
            current, path = queue.popleft()
            if len(path) > max_depth + 1:
                continue

            callees = graph.get(current, set())
            for callee in callees:
                if callee in visited:
                    continue
                visited.add(callee)
                new_path = path + [callee]

                # Check if callee name matches any HCCL-related pattern
                if self._name_is_hccl_related(callee):
                    return self._build_tier3_result(api, new_path)

                # Check if callee exists in a file that has HCCL references
                if callee in self._hccl_func_names:
                    return self._build_tier3_result(api, new_path)

                if len(new_path) <= max_depth + 1:
                    queue.append((callee, new_path))

        return None

    @staticmethod
    def _name_is_hccl_related(name: str) -> bool:
        lowered = name.lower()
        return any(kw in lowered for kw in (
            "hccl", "hcom", "npu", "processgroup",
            "allreduce", "allgather", "reduce_scatter",
        ))

    @staticmethod
    def _build_tier3_result(
        api: DistributedAPI, path: Sequence[str]
    ) -> AnalysisResult:
        chain = [
            DependencyEdge(
                caller=path[i],
                callee=path[i + 1],
                file=api.file,
                line=0,
            )
            for i in range(len(path) - 1)
        ]
        ref = HCCLReference(
            file=api.file,
            line=0,
            pattern=f"call-chain:{' -> '.join(path)}",
            context="(inferred via BFS)",
        )
        return AnalysisResult(
            api=api,
            confidence=Confidence.LOW,
            references=[ref],
            chain=chain,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_source(self, filepath: str) -> str | None:
        if filepath in self._file_sources:
            return self._file_sources[filepath]
        try:
            text = Path(filepath).read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = None
        self._file_sources[filepath] = text  # type: ignore[assignment]
        return text

    def _get_call_graph(self, filepath: str) -> dict[str, set[str]]:
        if filepath not in self._call_graphs:
            self._call_graphs[filepath] = extract_function_calls(Path(filepath))
        return self._call_graphs[filepath]
