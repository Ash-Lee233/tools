"""Three-tier HCCL dependency analysis engine.

Analysis is **function-scoped**: for each API we locate its actual function
body (resolving monkey-patch targets) and check only *that* body for HCCL
patterns and ProcessGroup dispatch calls.  This avoids false positives from
unrelated code in the same file.
"""

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
    get_function_body_ranges,
    iter_python_files,
    scan_hccl_in_range,
    scan_hccl_references,
)

# ---------------------------------------------------------------------------
# ProcessGroup dispatch patterns (Tier 2)
# ---------------------------------------------------------------------------

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

_NPU_BACKEND_RE = re.compile(
    r"""_get_backend\s*\(\s*torch\.device\s*\(\s*['"]npu['"]\s*\)\s*\)"""
)


def _scan_pg_dispatch_in_lines(
    lines: list[str], offset: int = 1,
) -> list[tuple[int, str, str]]:
    """Return ``(lineno, callee, line_text)`` for PG dispatch within *lines*.

    *offset* is the 1-based line number of the first line in the list.
    """
    hits: list[tuple[int, str, str]] = []
    for idx, line in enumerate(lines):
        lineno = offset + idx
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
        # caches
        self._hccl_by_file: dict[str, list[HCCLReference]] = {}
        self._apis: list[DistributedAPI] = []
        self._call_graphs: dict[str, dict[str, set[str]]] = {}
        self._file_sources: dict[str, list[str]] = {}   # file -> lines
        self._func_ranges: dict[str, dict[str, tuple[int, int]]] = {}
        self._hccl_files: set[str] = set()
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

        order = {Confidence.HIGH: 0, Confidence.MEDIUM: 1, Confidence.LOW: 2}
        results.sort(key=lambda r: (order[r.confidence], r.api.qualified_name))

        return AnalysisReport(
            repo_path=str(self.repo_path),
            total_apis=len(self._apis),
            results=results,
        )

    # ------------------------------------------------------------------
    # Phase 0: repository scan
    # ------------------------------------------------------------------

    def _scan_repo(self) -> None:
        for filepath in iter_python_files(self.repo_path):
            fstr = str(filepath)

            # HCCL references (file-level, for Tier 3 cross-file lookup)
            refs = scan_hccl_references(filepath)
            if refs:
                self._hccl_by_file[fstr] = refs
                self._hccl_files.add(fstr)

            # Public APIs
            self._apis.extend(extract_public_apis(filepath, self.repo_path))

            # Monkey-patches
            self._apis.extend(extract_monkey_patches(filepath))

        # Pre-compute function ranges for all HCCL-containing files +
        # all distributed files (needed for monkey-patch resolution)
        files_to_index: set[str] = set(self._hccl_files)
        for api in self._apis:
            files_to_index.add(api.file)
        for fstr in files_to_index:
            self._get_func_ranges(fstr)

        # Cross-file function names from HCCL files (for Tier 3)
        for fstr in self._hccl_files:
            graph = self._get_call_graph(fstr)
            self._hccl_func_names.update(graph.keys())

    # ------------------------------------------------------------------
    # Per-API analysis
    # ------------------------------------------------------------------

    def _analyse_api(self, api: DistributedAPI) -> AnalysisResult | None:
        # Resolve the actual implementation location
        impl_file, impl_func, impl_range = self._resolve_target(api)

        # Tier 1: direct HCCL reference in function body
        t1 = self._tier1(api, impl_file, impl_func, impl_range)
        if t1 is not None:
            return t1

        # Tier 2: PG dispatch in function body
        t2 = self._tier2(api, impl_file, impl_func, impl_range)
        if t2 is not None:
            return t2

        # Tier 3: BFS multi-hop from function
        t3 = self._tier3(api, impl_file, impl_func)
        if t3 is not None:
            return t3

        return None

    # ------------------------------------------------------------------
    # Target resolution (monkey-patch → actual function)
    # ------------------------------------------------------------------

    def _resolve_target(
        self, api: DistributedAPI,
    ) -> tuple[str, str, tuple[int, int] | None]:
        """Return ``(file, func_name, (start, end) | None)`` for the impl."""
        # Determine function name from the API qualified name
        func_name = api.qualified_name.rsplit(".", 1)[-1]

        if api.kind == "monkey-patch" and api.impl_target:
            resolved = self._resolve_impl_path(api.impl_target)
            if resolved is not None:
                rfile, rfunc = resolved
                rng = self._get_func_ranges(rfile).get(rfunc)
                if rng is not None:
                    return rfile, rfunc, rng

        # For regular APIs: look up in the API's own file
        rng = self._get_func_ranges(api.file).get(func_name)
        return api.file, func_name, rng

    def _resolve_impl_path(self, impl_target: str) -> tuple[str, str] | None:
        """Resolve ``'torch_npu.distributed.distributed_c10d._gather'`` →
        ``(filepath, '_gather')``."""
        parts = impl_target.split(".")
        if len(parts) < 2:
            return None

        func_name = parts[-1]
        mod_parts = parts[:-1]

        # Try as a module .py file
        mod_path = self.repo_path / Path(*mod_parts).with_suffix(".py")
        if mod_path.is_file():
            fstr = str(mod_path)
            if func_name in self._get_func_ranges(fstr):
                return fstr, func_name

        # Try as package __init__.py
        init_path = self.repo_path / Path(*mod_parts) / "__init__.py"
        if init_path.is_file():
            fstr = str(init_path)
            if func_name in self._get_func_ranges(fstr):
                return fstr, func_name

        # Fuzzy: search all indexed files for a function with this name
        for fstr, ranges in self._func_ranges.items():
            if func_name in ranges:
                return fstr, func_name

        return None

    # ------------------------------------------------------------------
    # Tier 1 – direct HCCL reference in function body
    # ------------------------------------------------------------------

    def _tier1(
        self,
        api: DistributedAPI,
        impl_file: str,
        impl_func: str,
        impl_range: tuple[int, int] | None,
    ) -> AnalysisResult | None:
        if impl_range is None:
            return None

        refs = scan_hccl_in_range(
            Path(impl_file), impl_range[0], impl_range[1],
        )
        if not refs:
            return None

        chain = self._build_chain_prefix(api, impl_file, impl_func, impl_range)
        return AnalysisResult(
            api=api,
            confidence=Confidence.HIGH,
            references=refs,
            chain=chain,
        )

    # ------------------------------------------------------------------
    # Tier 2 – ProcessGroup dispatch in function body
    # ------------------------------------------------------------------

    def _tier2(
        self,
        api: DistributedAPI,
        impl_file: str,
        impl_func: str,
        impl_range: tuple[int, int] | None,
    ) -> AnalysisResult | None:
        if impl_range is None:
            return None

        lines = self._read_lines(impl_file)
        if not lines:
            return None

        start, end = impl_range
        body = lines[start - 1 : end]
        hits = _scan_pg_dispatch_in_lines(body, offset=start)
        if not hits:
            return None

        refs = [
            HCCLReference(
                file=impl_file, line=ln,
                pattern=f"PG.{callee}", context=ctx,
            )
            for ln, callee, ctx in hits
        ]
        chain = self._build_chain_prefix(api, impl_file, impl_func, impl_range)
        for ln, callee, _ctx in hits:
            chain.append(DependencyEdge(
                caller=impl_func,
                callee=f"ProcessGroup.{callee}",
                file=impl_file,
                line=ln,
            ))
            chain.append(DependencyEdge(
                caller=f"ProcessGroup.{callee}",
                callee=f"ProcessGroupHCCL.{callee}",
                file="(runtime dispatch)",
                line=0,
            ))
            break  # one dispatch edge is enough
        return AnalysisResult(
            api=api,
            confidence=Confidence.MEDIUM,
            references=refs,
            chain=chain,
        )

    # ------------------------------------------------------------------
    # Tier 3 – BFS call-chain (max depth 3)
    # ------------------------------------------------------------------

    def _tier3(
        self,
        api: DistributedAPI,
        impl_file: str,
        impl_func: str,
        max_depth: int = 3,
    ) -> AnalysisResult | None:
        graph = self._get_call_graph(impl_file)
        if not graph:
            return None

        # Also try underscore variants
        start = None
        for candidate in (impl_func, "_" + impl_func, impl_func.lstrip("_")):
            if candidate in graph:
                start = candidate
                break
        if start is None:
            return None

        queue: deque[tuple[str, list[str]]] = deque()
        queue.append((start, [start]))
        visited: set[str] = {start}

        while queue:
            current, path = queue.popleft()
            if len(path) > max_depth + 1:
                continue

            for callee in graph.get(current, set()):
                if callee in visited:
                    continue
                visited.add(callee)
                new_path = path + [callee]

                if self._name_is_hccl_related(callee) or callee in self._hccl_func_names:
                    return self._build_tier3_result(api, impl_file, impl_func, new_path)

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

    def _build_tier3_result(
        self,
        api: DistributedAPI,
        impl_file: str,
        impl_func: str,
        path: Sequence[str],
    ) -> AnalysisResult:
        chain = self._build_chain_prefix(api, impl_file, impl_func, None)
        for i in range(len(path) - 1):
            chain.append(DependencyEdge(
                caller=path[i], callee=path[i + 1],
                file=impl_file, line=0,
            ))
        ref = HCCLReference(
            file=impl_file, line=0,
            pattern=f"call-chain:{' -> '.join(path)}",
            context="(inferred via BFS)",
        )
        return AnalysisResult(
            api=api, confidence=Confidence.LOW,
            references=[ref], chain=chain,
        )

    # ------------------------------------------------------------------
    # Chain helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_chain_prefix(
        api: DistributedAPI,
        impl_file: str,
        impl_func: str,
        impl_range: tuple[int, int] | None,
    ) -> list[DependencyEdge]:
        """If the API is a monkey-patch pointing to a different file/function,
        create a leading edge ``API → impl_func``."""
        if api.kind != "monkey-patch":
            return []
        if api.file == impl_file:
            return []
        return [DependencyEdge(
            caller=api.qualified_name,
            callee=f"{impl_func}()",
            file=impl_file,
            line=impl_range[0] if impl_range else 0,
        )]

    # ------------------------------------------------------------------
    # Caches
    # ------------------------------------------------------------------

    def _read_lines(self, filepath: str) -> list[str]:
        if filepath not in self._file_sources:
            try:
                text = Path(filepath).read_text(encoding="utf-8", errors="replace")
                self._file_sources[filepath] = text.splitlines()
            except OSError:
                self._file_sources[filepath] = []
        return self._file_sources[filepath]

    def _get_func_ranges(self, filepath: str) -> dict[str, tuple[int, int]]:
        if filepath not in self._func_ranges:
            self._func_ranges[filepath] = get_function_body_ranges(Path(filepath))
        return self._func_ranges[filepath]

    def _get_call_graph(self, filepath: str) -> dict[str, set[str]]:
        if filepath not in self._call_graphs:
            self._call_graphs[filepath] = extract_function_calls(Path(filepath))
        return self._call_graphs[filepath]
