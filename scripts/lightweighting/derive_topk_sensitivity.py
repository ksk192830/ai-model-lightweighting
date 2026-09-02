#!/usr/bin/env python3
"""Derive a reproducible top-k mixed-precision policy from a sensitivity map."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    args = parser.parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")

    source_path = resolve(args.input)
    report = json.loads(source_path.read_text(encoding="utf-8"))
    blocks = report.get("blocks", {})
    if not isinstance(blocks, dict) or not blocks:
        raise ValueError("Sensitivity report has no blocks")
    if all(isinstance(value, dict) and "drop_max" in value for value in blocks.values()):
        ranked = sorted(blocks, key=lambda name: (-float(blocks[name]["drop_max"]), name))
        metric = "drop_max"
    elif all(isinstance(value, dict) and "kld_total" in value for value in blocks.values()):
        ranked = sorted(blocks, key=lambda name: (-float(blocks[name]["kld_total"]), name))
        metric = "kld_total"
    else:
        raise ValueError("Sensitivity blocks lack a common ranking metric")

    report["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["derived_from"] = str(source_path.relative_to(ROOT))
    report["selection_rule"] = {"type": "top-k", "k": args.top_k, "metric": metric}
    report["ranked_blocks"] = ranked
    report["sensitive_blocks"] = ranked[: args.top_k]
    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"top-{args.top_k}: {report['sensitive_blocks']}")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
