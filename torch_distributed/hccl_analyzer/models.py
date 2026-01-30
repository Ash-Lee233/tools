"""Data structures for HCCL dependency analysis."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class Confidence(enum.Enum):
    """Confidence tier for an HCCL dependency."""

    HIGH = "high"      # Tier 1: direct HCCL reference in same file
    MEDIUM = "medium"  # Tier 2: via ProcessGroup abstraction
    LOW = "low"        # Tier 3: multi-hop call chain (BFS depth <= 3)


@dataclass
class HCCLReference:
    """A single occurrence of an HCCL-related pattern in source code."""

    file: str
    line: int
    pattern: str   # the matched pattern string
    context: str   # the source line containing the match


@dataclass
class DistributedAPI:
    """A public API discovered under ``torch.distributed``."""

    qualified_name: str   # e.g. "torch.distributed.all_reduce"
    file: str
    line: int
    kind: str = "function"  # function | class | method


@dataclass
class DependencyEdge:
    """One hop in a dependency chain from an API to an HCCL reference."""

    caller: str   # qualified name or file:line
    callee: str
    file: str
    line: int


@dataclass
class AnalysisResult:
    """Result for a single API that depends on HCCL."""

    api: DistributedAPI
    confidence: Confidence
    references: list[HCCLReference] = field(default_factory=list)
    chain: list[DependencyEdge] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "api": self.api.qualified_name,
            "file": self.api.file,
            "line": self.api.line,
            "kind": self.api.kind,
            "confidence": self.confidence.value,
            "references": [
                {"file": r.file, "line": r.line, "pattern": r.pattern, "context": r.context}
                for r in self.references
            ],
            "chain": [
                {"caller": e.caller, "callee": e.callee, "file": e.file, "line": e.line}
                for e in self.chain
            ],
        }


@dataclass
class AnalysisReport:
    """Complete analysis report."""

    repo_path: str
    total_apis: int = 0
    results: list[AnalysisResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = {"high": [], "medium": [], "low": []}
        for r in self.results:
            grouped[r.confidence.value].append(r.to_dict())
        return {
            "repo_path": self.repo_path,
            "total_apis_scanned": self.total_apis,
            "hccl_dependent_apis": len(self.results),
            "by_confidence": grouped,
        }

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append("=" * 70)
        lines.append("HCCL Dependency Analysis Report")
        lines.append("=" * 70)
        lines.append(f"Repository : {self.repo_path}")
        lines.append(f"Total APIs scanned : {self.total_apis}")
        lines.append(f"HCCL-dependent APIs: {len(self.results)}")
        lines.append("")

        for tier, label in [
            (Confidence.HIGH, "Tier 1 - HIGH confidence (direct HCCL reference)"),
            (Confidence.MEDIUM, "Tier 2 - MEDIUM confidence (via ProcessGroup abstraction)"),
            (Confidence.LOW, "Tier 3 - LOW confidence (multi-hop call chain)"),
        ]:
            items = [r for r in self.results if r.confidence == tier]
            lines.append("-" * 70)
            lines.append(f"{label}  [{len(items)} APIs]")
            lines.append("-" * 70)
            if not items:
                lines.append("  (none)")
                lines.append("")
                continue
            for r in items:
                lines.append(f"  {r.api.qualified_name}")
                lines.append(f"    defined at {r.api.file}:{r.api.line}  ({r.api.kind})")
                if r.references:
                    lines.append("    HCCL references:")
                    for ref in r.references:
                        lines.append(f"      - {ref.file}:{ref.line}  pattern={ref.pattern!r}")
                        lines.append(f"        {ref.context.strip()}")
                if r.chain:
                    lines.append("    call chain:")
                    for edge in r.chain:
                        lines.append(f"      {edge.caller} -> {edge.callee}  ({edge.file}:{edge.line})")
                lines.append("")

        return "\n".join(lines)
