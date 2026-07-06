from __future__ import annotations

import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))

from benchmark import CSV_FIELDS, write_results_csv  # noqa: E402


def test_write_results_csv_preserves_appended_metric_columns(tmp_path: Path) -> None:
    output = tmp_path / "metrics.csv"
    extended_fields = [*CSV_FIELDS, "Mask AP", "Mask mIoU"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=extended_fields)
        writer.writeheader()
        writer.writerow({"Model Path": "existing.engine", "Mask AP": "0.5"})

    write_results_csv(output, [{"Model Path": "new.engine", "FPS": 32.1}])

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["Mask AP"] == "0.5"
    assert rows[1]["Model Path"] == "new.engine"
    assert rows[1]["FPS"] == "32.100000"
    assert rows[1]["Mask AP"] == ""
    assert rows[1]["Mask mIoU"] == ""
