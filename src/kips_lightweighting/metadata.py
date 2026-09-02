"""Metadata helpers shared by experiment entry points."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .registry import REPOSITORY_ROOT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def runtime_metadata() -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git_commit": git_commit(),
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def refresh_experiment_documents(*experiment_ids: str) -> None:
    """Synchronize canonical metadata and the generated artifact index."""
    if os.environ.get("KIPS_SKIP_DOCUMENT_REFRESH", "").lower() in {
        "1",
        "true",
        "yes",
    }:
        print("experiment document refresh skipped by environment")
        return
    sync_command = [
        sys.executable,
        str(
            REPOSITORY_ROOT
            / "scripts"
            / "experiments"
            / "sync_metadata.py"
        ),
        *experiment_ids,
    ]
    index_command = [
        sys.executable,
        str(
            REPOSITORY_ROOT
            / "scripts"
            / "experiments"
            / "update_index.py"
        ),
    ]
    subprocess.run(sync_command, cwd=REPOSITORY_ROOT, check=True)
    subprocess.run(index_command, cwd=REPOSITORY_ROOT, check=True)
