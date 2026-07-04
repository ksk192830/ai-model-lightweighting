from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate import (  # noqa: E402
    DatasetSample,
    GroundTruth,
    Prediction,
    UltralyticsPredictor,
    evaluate_predictions,
)


def make_sample() -> DatasetSample:
    image_id = str(Path("/tmp/test-image.png").resolve())
    target = GroundTruth(
        image_id=image_id,
        class_id=0,
        box=(10.0, 10.0, 30.0, 30.0),
    )
    return DatasetSample(
        image_id=image_id,
        image_path=Path(image_id),
        width=100,
        height=100,
        targets=(target,),
    )


def test_coco_metrics_match_perfect_prediction() -> None:
    sample = make_sample()
    prediction = Prediction(
        image_id=sample.image_id,
        class_id=0,
        confidence=0.9,
        box=(10.0, 10.0, 30.0, 30.0),
    )

    metrics = evaluate_predictions(
        [sample],
        [prediction],
        metric_backend="coco",
        operating_conf=0.25,
    )

    assert metrics["map50"] == pytest.approx(1.0)
    assert metrics["map50_95"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)


def test_operating_conf_does_not_truncate_coco_ap_curve() -> None:
    sample = make_sample()
    predictions = [
        Prediction(
            image_id=sample.image_id,
            class_id=0,
            confidence=0.9,
            box=(10.0, 10.0, 30.0, 30.0),
        ),
        Prediction(
            image_id=sample.image_id,
            class_id=0,
            confidence=0.01,
            box=(50.0, 50.0, 70.0, 70.0),
        ),
    ]

    metrics = evaluate_predictions(
        [sample],
        predictions,
        metric_backend="coco",
        operating_conf=0.25,
    )

    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["num_predictions"] == 2.0


def test_ultralytics_classes_align_to_coco_names_with_background_category() -> None:
    predictor = UltralyticsPredictor.__new__(UltralyticsPredictor)
    predictor.class_names = {
        0: "out_line",
        1: "parking_lot",
        2: "parking_space",
    }
    predictor.class_id_map = {}
    sample = DatasetSample(
        image_id="/tmp/image.png",
        image_path=Path("/tmp/image.png"),
        width=100,
        height=100,
        targets=(
            GroundTruth(
                image_id="/tmp/image.png",
                class_id=1,
                box=(0, 0, 10, 10),
                class_name="out_line",
            ),
            GroundTruth(
                image_id="/tmp/image.png",
                class_id=2,
                box=(20, 20, 30, 30),
                class_name="parking_lot",
            ),
            GroundTruth(
                image_id="/tmp/image.png",
                class_id=3,
                box=(40, 40, 50, 50),
                class_name="parking_space",
            ),
        ),
    )

    mapping = predictor.align_classes([sample])

    assert mapping == {0: 1, 1: 2, 2: 3}
