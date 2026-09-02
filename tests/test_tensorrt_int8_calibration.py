from __future__ import annotations

import importlib.util
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = REPOSITORY_ROOT / "scripts/lightweighting/build_tensorrt_int8.py"
    spec = importlib.util.spec_from_file_location("build_tensorrt_int8", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calibration_selection_is_deterministic_and_bounded() -> None:
    module = load_module()
    images = [Path(f"image-{index:03d}.jpg") for index in range(200)]
    first = module.select_calibration_images(images, 128, 42)
    second = module.select_calibration_images(list(reversed(images)), 128, 42)

    assert first == second
    assert len(first) == 128
    assert len(set(first)) == 128
    assert set(first).issubset(images)


def test_calibration_selection_rejects_invalid_count() -> None:
    module = load_module()
    images = [Path("one.jpg")]

    for count in (0, 2):
        try:
            module.select_calibration_images(images, count, 42)
        except ValueError:
            pass
        else:
            raise AssertionError(f"count={count} should fail")
