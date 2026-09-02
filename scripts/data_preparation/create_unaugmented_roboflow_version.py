#!/usr/bin/env python3
"""Generate and download an augmentation-free Roboflow dataset version."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "data" / "training" / "front_unaugmented"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="hmobility-fvu9x")
    parser.add_argument("--project", default="parking_front")
    parser.add_argument("--source-version", type=int, default=8)
    parser.add_argument("--format", default="coco-segmentation")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--api-key-env",
        default="ROBOFLOW_API_KEY",
        help="Read the API key from this environment variable; prompt if absent.",
    )
    parser.add_argument("--version-wait-seconds", type=int, default=1800)
    parser.add_argument(
        "--gui-prompt",
        action="store_true",
        help="Request the API key in a local password dialog instead of the terminal.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def wait_for_version(project, version_number: int, timeout_seconds: int):
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return project.version(version_number)
        except RuntimeError as error:
            last_error = error
            time.sleep(10)
    raise TimeoutError(
        f"Roboflow version {version_number} did not become visible within "
        f"{timeout_seconds} seconds: {last_error}"
    )


def main() -> int:
    args = parse_args()
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        if args.gui_prompt:
            response = subprocess.run(
                [
                    "zenity",
                    "--password",
                    "--title=Roboflow 인증",
                    "--text=Roboflow API key를 입력하세요. 입력값은 저장되지 않습니다.",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if response.returncode != 0:
                raise RuntimeError("Roboflow API key input was cancelled.")
            api_key = response.stdout.strip()
        else:
            api_key = getpass.getpass("Roboflow API key (input hidden): ").strip()
    if not api_key:
        raise ValueError("A Roboflow API key is required.")

    from roboflow import Roboflow

    client = Roboflow(api_key=api_key)
    project = client.workspace(args.workspace).project(args.project)
    source_version = project.version(args.source_version)
    source_preprocessing = dict(source_version.preprocessing or {})
    source_augmentation = dict(source_version.augmentation or {})
    settings = {
        "preprocessing": source_preprocessing,
        "augmentation": {},
    }

    print(
        json.dumps(
            {
                "action": "generate augmentation-free Roboflow version",
                "workspace": args.workspace,
                "project": args.project,
                "source_version": args.source_version,
                "source_preprocessing": source_preprocessing,
                "source_augmentation": source_augmentation,
                "new_augmentation": {},
                "output": str(output),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    new_version_number = project.generate_version(settings=settings)
    new_version = wait_for_version(
        project,
        new_version_number,
        args.version_wait_seconds,
    )
    dataset = new_version.download(
        args.format,
        location=str(output),
        overwrite=False,
    )

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "roboflow",
        "workspace": args.workspace,
        "project": args.project,
        "source_version": args.source_version,
        "generated_version": int(new_version_number),
        "format": args.format,
        "preprocessing": dict(new_version.preprocessing or {}),
        "augmentation": dict(new_version.augmentation or {}),
        "source_augmentation": source_augmentation,
        "roboflow_images": new_version.images,
        "roboflow_splits": new_version.splits,
        "download_location": str(Path(dataset.location).resolve()),
        "augmentation_free": not bool(new_version.augmentation),
    }
    if not manifest["augmentation_free"]:
        raise RuntimeError(
            "The generated Roboflow version still reports augmentation settings."
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "roboflow_export_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"downloaded augmentation-free export: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
