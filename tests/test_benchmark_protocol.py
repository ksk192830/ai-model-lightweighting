from __future__ import annotations

import importlib.util
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts/evaluation/benchmark_baseline.py"
    spec = importlib.util.spec_from_file_location("benchmark_baseline_protocol", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_multi_image_selection_is_seeded_and_ignores_non_images(
    tmp_path: Path,
) -> None:
    for index in range(6):
        Image.new("RGB", (4, 4), (index, 0, 0)).save(tmp_path / f"{index}.png")
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")
    module = load_module()

    first = module.image_files(tmp_path, 4, 42)
    second = module.image_files(tmp_path, 4, 42)
    different = module.image_files(tmp_path, 4, 7)

    assert first == second
    assert first != different
    assert len(first) == 4
    assert all(path.suffix == ".png" for path in first)
