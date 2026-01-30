"""Command-line interface for the HCCL dependency analyzer."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from .analyzer import HCCLAnalyzer


def _clone_repo(url: str, dest: Path) -> None:
    print(f"Cloning {url} into {dest} ...")
    subprocess.check_call(
        ["git", "clone", "--depth", "1", url, str(dest)],
        stdout=sys.stderr,
        stderr=sys.stderr,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hccl-analyzer",
        description="Analyse torch.distributed APIs for HCCL dependencies.",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--repo-path",
        type=str,
        help="Path to a local Ascend PyTorch repository.",
    )
    src.add_argument(
        "--repo-url",
        type=str,
        help="Git URL to clone (shallow) before analysis.",
    )
    p.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        default=False,
        help="Emit structured JSON instead of human-readable text.",
    )
    p.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Write output to this file instead of stdout.",
    )
    p.add_argument(
        "--excel",
        type=str,
        default=None,
        metavar="FILE",
        help="Export results to an Excel (.xlsx) file.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Resolve repository path
    tmp_dir = None
    if args.repo_url:
        tmp_dir = tempfile.mkdtemp(prefix="hccl_analyzer_")
        repo_path = Path(tmp_dir) / "repo"
        _clone_repo(args.repo_url, repo_path)
    else:
        repo_path = Path(args.repo_path)

    if not repo_path.is_dir():
        parser.error(f"Repository path does not exist: {repo_path}")

    # Run analysis
    analyzer = HCCLAnalyzer(repo_path)
    report = analyzer.run()

    # Excel export
    if args.excel:
        report.export_excel(args.excel)
        print(f"Excel report written to {args.excel}", file=sys.stderr)

    # Format output
    if args.json_output:
        text = json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
    else:
        text = report.format_text()

    # Write output
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        if not args.excel:
            print(text)
