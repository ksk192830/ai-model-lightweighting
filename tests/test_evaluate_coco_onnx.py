import json
import sys
from pathlib import Path

import pytest


sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "scripts" / "evaluation"),
)

from benchmark_baseline import postprocess_num_select  # noqa: E402
from evaluate_coco_rfdetr_onnx import (  # noqa: E402
    accuracy_parity,
    recheck_accuracy_parity,
    sha256,
)


def metrics(bbox: float, mask: float, miou: float) -> dict:
    return {
        "bbox": {"ap": bbox},
        "segm": {"ap": mask},
        "semantic_miou": miou,
    }


def test_sha256_streams_file(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"abc")

    assert sha256(path) == (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )


def test_tensorrt_postprocess_uses_static_query_count() -> None:
    assert postprocess_num_select((1, 200, 4)) == 200

    with pytest.raises(ValueError, match="static batch-1"):
        postprocess_num_select((2, 200, 4))


def test_accuracy_parity_checks_protocol_and_metric_tolerances(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(
        json.dumps(
            {
                "image_count": 437,
                "threshold": 0.001,
                "miou_threshold": 0.25,
                "metrics": metrics(0.7, 0.6, 0.5),
            }
        ),
        encoding="utf-8",
    )

    parity = accuracy_parity(
        metrics(0.70001, 0.59999, 0.50001),
        reference_path,
        candidate_prediction_count=100,
        image_count=437,
        threshold=0.001,
        miou_threshold=0.25,
        max_absolute_ap_delta=1.0e-4,
        max_absolute_miou_delta=1.0e-4,
    )

    assert parity["passed"] is True
    assert parity["protocol_checks"] == {
        "image_count": True,
        "threshold": True,
        "miou_threshold": True,
    }
    assert parity["delta_absolute"]["bbox_ap"] == pytest.approx(1.0e-5)


def test_accuracy_parity_fails_on_protocol_mismatch(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(
        json.dumps(
            {
                "image_count": 1,
                "threshold": 0.001,
                "miou_threshold": 0.25,
                "metrics": metrics(0.7, 0.6, 0.5),
            }
        ),
        encoding="utf-8",
    )

    parity = accuracy_parity(
        metrics(0.7, 0.6, 0.5),
        reference_path,
        candidate_prediction_count=100,
        image_count=437,
        threshold=0.001,
        miou_threshold=0.25,
        max_absolute_ap_delta=1.0e-4,
        max_absolute_miou_delta=1.0e-4,
    )

    assert parity["passed"] is False
    assert parity["protocol_checks"]["image_count"] is False


def test_recheck_changes_only_parity_metadata(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(
        json.dumps(
            {
                "image_count": 437,
                "prediction_count": 100,
                "threshold": 0.001,
                "miou_threshold": 0.25,
                "metrics": metrics(0.7, 0.6, 0.5),
            }
        ),
        encoding="utf-8",
    )
    result_path = tmp_path / "onnx.json"
    original_metrics = metrics(0.7002, 0.6001, 0.5003)
    result_path.write_text(
        json.dumps(
            {
                "image_count": 437,
                "prediction_count": 99,
                "threshold": 0.001,
                "miou_threshold": 0.25,
                "timing": {"wall_seconds": 123.0},
                "metrics": original_metrics,
                "pytorch_parity": {"passed": False},
            }
        ),
        encoding="utf-8",
    )

    refreshed = recheck_accuracy_parity(
        result_path,
        reference_path,
        max_absolute_ap_delta=0.005,
        max_absolute_miou_delta=0.005,
    )

    assert refreshed["metrics"] == original_metrics
    assert refreshed["timing"] == {"wall_seconds": 123.0}
    assert refreshed["pytorch_parity"]["passed"] is True
    assert refreshed["pytorch_parity"]["prediction_count"]["delta"] == -1
