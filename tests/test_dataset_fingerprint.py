from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts/data_preparation/fingerprint_coco_dataset.py"
    spec = importlib.util.spec_from_file_location("fingerprint_coco_dataset", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_dataset(path: Path, *, reordered: bool = False, x: float = 1.0) -> None:
    path.mkdir()
    (path / "a.jpg").write_bytes(b"first-image")
    (path / "b.jpg").write_bytes(b"second-image")
    images = [
        {"id": 10, "file_name": "a.jpg", "width": 4, "height": 3},
        {"id": 20, "file_name": "b.jpg", "width": 4, "height": 3},
    ]
    annotations = [
        {
            "id": 100,
            "image_id": 10,
            "category_id": 1,
            "bbox": [x, 1, 2, 2],
            "segmentation": [[1, 1, 3, 1, 3, 3]],
            "area": 4,
            "iscrowd": 0,
        }
    ]
    if reordered:
        images.reverse()
        annotations[0]["id"] = 999
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "parking", "supercategory": "none"}],
    }
    kwargs = {"indent": 2} if reordered else {"separators": (",", ":")}
    (path / "_annotations.coco.json").write_text(
        json.dumps(coco, **kwargs), encoding="utf-8"
    )


def test_reordered_json_and_annotation_ids_are_equivalent(tmp_path: Path) -> None:
    module = load_module()
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_dataset(first)
    write_dataset(second, reordered=True)

    left = module.build_fingerprint(first, "desktop")
    right = module.build_fingerprint(second, "notebook")
    comparison = module.compare_reports(left, right)

    assert left["raw_annotation_sha256"] != right["raw_annotation_sha256"]
    assert comparison["equivalent"] is True
    assert comparison["raw_annotation_bytes_equal"] is False


def test_annotation_geometry_change_is_detected(tmp_path: Path) -> None:
    module = load_module()
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_dataset(first)
    write_dataset(second, x=1.5)

    comparison = module.compare_reports(
        module.build_fingerprint(first, "desktop"),
        module.build_fingerprint(second, "notebook"),
    )

    assert comparison["equivalent"] is False
    assert comparison["comparisons"]["coco_evaluation_semantics_sha256"] is False
    assert comparison["comparisons"]["image_collection_sha256"] is True


def test_image_content_change_is_detected(tmp_path: Path) -> None:
    module = load_module()
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_dataset(first)
    write_dataset(second)
    (second / "b.jpg").write_bytes(b"different-image")

    comparison = module.compare_reports(
        module.build_fingerprint(first, "desktop"),
        module.build_fingerprint(second, "notebook"),
    )

    assert comparison["equivalent"] is False
    assert comparison["comparisons"]["coco_evaluation_semantics_sha256"] is True
    assert comparison["comparisons"]["image_collection_sha256"] is False


def test_incomplete_reports_are_rejected() -> None:
    module = load_module()

    with pytest.raises(ValueError, match="lacks required fields"):
        module.compare_reports(
            {"schema_version": 1, "label": "desktop"},
            {"schema_version": 1, "label": "notebook"},
        )
