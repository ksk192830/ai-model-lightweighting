import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "scripts" / "evaluation"),
)

import evaluate_coco_tensorrt as evaluation  # noqa: E402


class SegmentationMetricTests(unittest.TestCase):
    def test_semantic_iou_is_category_mean(self) -> None:
        miou, category_ious = evaluation.semantic_iou(
            {1: 6, 2: 1},
            {1: 10, 2: 2},
        )

        self.assertAlmostEqual(miou, 0.55)
        self.assertEqual(category_ious, {1: 0.6, 2: 0.5})

    def test_csv_update_adds_columns_without_changing_existing_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / "artifacts/experiments/B01/front/model.engine"
            engine.parent.mkdir(parents=True)
            engine.touch()
            output = root / "results.csv"
            output.write_text(
                "Model Path,mAP50,Status\n"
                "artifacts/experiments/B01/front/model.engine,0.98,ok\n",
                encoding="utf-8",
            )

            with patch.object(evaluation, "REPOSITORY_ROOT", root):
                evaluation.update_metrics_csv(
                    output,
                    engine,
                    "B01",
                    {
                        "Mask AP": 0.7,
                        "Mask AP50": 0.9,
                        "Mask AP75": 0.8,
                        "Mask mIoU": 0.75,
                    },
                )

            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["mAP50"], "0.98")
            self.assertEqual(rows[0]["Status"], "ok")
            self.assertEqual(rows[0]["Mask AP"], "0.700000")
            self.assertEqual(rows[0]["Mask mIoU"], "0.750000")


if __name__ == "__main__":
    unittest.main()
