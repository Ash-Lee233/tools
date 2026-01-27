from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Dict, Set, Tuple

@dataclass
class SymbolResolution:
    """Resolution result for a dotted symbol."""
    symbol: str
    status: str  # "ok" | "miss"
    module: Optional[str] = None
    attrpath: Optional[List[str]] = None
    hint_modules: Optional[List[str]] = None
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "status": self.status,
            "module": self.module,
            "attrpath": self.attrpath,
            "hint_modules": self.hint_modules,
            "error": self.error,
        }

def _try_resolve(dotted: str) -> Optional[Tuple[str, List[str]]]:
    parts = dotted.split(".")
    # Try importing progressively shorter module prefixes
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
        if ok:
            return modname, list(attrpath)
    return None

def _scan_torch_exports(prefixes: Sequence[str]) -> Dict[str, Set[str]]:
    """Build a reverse index leaf_name -> modules that export it in __all__.

    This is best-effort: many internal modules don't set __all__.
    """
    try:
        import torch  # noqa: F401
    except Exception:
        return {}

    import torch as _torch

    hits: Dict[str, Set[str]] = {}
    for m in pkgutil.walk_packages(_torch.__path__, _torch.__name__ + "."):
        name = m.name
        if not any(name.startswith(p) for p in prefixes):
            continue
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        exported = getattr(mod, "__all__", None)
        if not exported:
            continue
        for k in exported:
            hits.setdefault(k, set()).add(name)
    return hits

def resolve_symbols(
    symbols: Iterable[str],
    *,
    with_hints: bool = True,
    hint_prefixes: Optional[Sequence[str]] = None,
) -> List[SymbolResolution]:
    """Resolve dotted symbols against the current Python environment.

    Args:
        symbols: Iterable of dotted symbol strings.
        with_hints: If True, also scans torch.* modules' __all__ for hint modules.
        hint_prefixes: Which torch.* prefixes to scan for hints.

    Returns:
        List of SymbolResolution entries (same order as input).
    """
    hint_prefixes = hint_prefixes or (
        "torch._dynamo",
        "torch._inductor",
        "torch.distributed",
        "torch.cuda",
        "torch.accelerator",
        "torch.nn",
        "torch.optim",
        "torch.multiprocessing",
        "torch.nested",
    )

    hint_index: Dict[str, Set[str]] = _scan_torch_exports(hint_prefixes) if with_hints else {}

    results: List[SymbolResolution] = []
    for s in symbols:
        try:
            r = _try_resolve(s)
            if r:
                modname, attrpath = r
                results.append(SymbolResolution(symbol=s, status="ok", module=modname, attrpath=attrpath))
            else:
                leaf = s.split(".")[-1]
                hints = sorted(hint_index.get(leaf, set())) if with_hints else []
                results.append(SymbolResolution(symbol=s, status="miss", hint_modules=hints or None))
        except Exception as e:
            # Defensive: never crash; mark miss with error
            leaf = s.split(".")[-1]
            hints = sorted(hint_index.get(leaf, set())) if with_hints else []
            results.append(SymbolResolution(symbol=s, status="miss", hint_modules=hints or None, error=str(e)))
    return results
