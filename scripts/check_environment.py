#!/usr/bin/env python3
"""Verify the baseline RF-DETR runtime and checkpoint metadata."""

from __future__ import annotations

import sys
from importlib.metadata import version
from pathlib import Path

import torch
from rfdetr import RFDETRSegLarge  # noqa: F401


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSIONS = {
    "rfdetr": "1.8.1",
    "torch": "2.12.1",
    "torchvision": "0.27.1",
}
CHECKPOINTS = (
    REPOSITORY_ROOT / "models" / "parking_front.pth",
    REPOSITORY_ROOT / "models" / "parking_rear.pth",
)


def main() -> int:
    failed = False

    print(f"Python: {sys.version.split()[0]}")
    for package, expected in EXPECTED_VERSIONS.items():
        installed = version(package)
        status = "OK" if installed == expected else f"expected {expected}"
        print(f"{package}: {installed} ({status})")
        failed |= installed != expected

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"inference device: {device}")

    for checkpoint in CHECKPOINTS:
        if not checkpoint.is_file():
            print(f"{checkpoint.name}: missing")
            failed = True
            continue

        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model_name = payload.get("model_name")
        rfdetr_version = payload.get("rfdetr_version")
        valid = model_name == "RFDETRSegLarge" and rfdetr_version == "1.8.1"
        status = "OK" if valid else "metadata mismatch"
        print(
            f"{checkpoint.name}: model={model_name}, "
            f"rfdetr={rfdetr_version} ({status})"
        )
        failed |= not valid

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
