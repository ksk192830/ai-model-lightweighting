"""Tests for the read-only aggregate lightweighting pipeline monitor."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_monitor_module():
    path = REPOSITORY_ROOT / "scripts/training/watch_lightweighting_pipeline.py"
    spec = importlib.util.spec_from_file_location(
        "watch_lightweighting_pipeline", path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


monitor = load_monitor_module()


class LightweightingPipelineMonitorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_missing_files_are_reported_as_pending(self) -> None:
        output = monitor.render(self.root, processes=[], gpu="unavailable")

        self.assertIn("S02 training  : pending (not started)", output)
        self.assertIn("M01 training  : pending (not started)", output)
        self.assertIn("S02 CPU queue : pending", output)
        self.assertIn("M01 CPU queue : pending", output)
        self.assertIn("notebook bundle : pending/incomplete", output)
        self.assertIn("bundle archive  : pending", output)

    def test_training_queue_cpu_job_and_delivery_are_summarized(self) -> None:
        recovery = self.root / "artifacts/experiments/S02/front/recovery"
        recovery.mkdir(parents=True)
        report = {
            "created_at_utc": "2026-09-02T12:00:00+00:00",
            "status": "running",
            "profile": {"epochs": 10},
        }
        (recovery / "recovery-training.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        fields = ["epoch", "step", "val/mAP_50_95", "val/segm_mAP_50_95"]
        with (recovery / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(
                {
                    "epoch": "0",
                    "step": "226",
                    "val/mAP_50_95": "0.71",
                    "val/segm_mAP_50_95": "0.60",
                }
            )
            writer.writerow({"epoch": "1", "step": "249"})

        queue_dir = self.root / "artifacts/experiments/S02/front"
        (queue_dir / "post-recovery-queue.json").write_text(
            json.dumps(
                {
                    "pid": 30,
                    "status": "waiting-for-training",
                    "stages": [
                        {"name": "finalize-recovery", "status": "pending"}
                    ],
                }
            ),
            encoding="utf-8",
        )
        m01_queue_dir = self.root / "artifacts/experiments/M01/front"
        m01_queue_dir.mkdir(parents=True)
        (m01_queue_dir / "post-recovery-queue.json").write_text(
            json.dumps(
                {
                    "pid": 31,
                    "status": "waiting-for-training",
                    "training_status": "missing",
                    "stages": [
                        {"name": "finalize-recovery", "status": "pending"},
                        {"name": "onnx-equivalence", "status": "pending"},
                        {"name": "pth-test-evaluation", "status": "pending"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        bundle = self.root / "delivery/notebook-front-ready4"
        bundle.mkdir(parents=True)
        (bundle / "manifest.json").write_text("{}", encoding="utf-8")
        (bundle / "data-manifest.json").write_text("{}", encoding="utf-8")
        archive = self.root / "delivery/notebook-front-ready4.tar.gz"
        archive.write_bytes(b"archive")

        now = time.time()
        processes = [
            monitor.ProcessRecord(
                10,
                ("python", "scripts/experiments/train_candidate.py", "S02"),
                now - 10,
            ),
            monitor.ProcessRecord(
                20,
                (
                    "/bin/bash",
                    "while kill -0 10; train_candidate.py M01 after S02 queue",
                ),
                now - 5,
            ),
            monitor.ProcessRecord(30, ("python", "queue.py"), now - 3),
            monitor.ProcessRecord(31, ("python", "queue.py"), now - 3),
            monitor.ProcessRecord(
                40,
                (
                    "python",
                    "scripts/evaluation/evaluate_coco_rfdetr_onnx.py",
                    "--name",
                    "S01-test",
                ),
                now - 2,
            ),
        ]

        output = monitor.render(self.root, processes=processes, gpu="util 50%")

        self.assertIn("S02 training  : running | running (PID 10)", output)
        self.assertIn("1/10 completed", output)
        self.assertIn("bbox 0.7100 | mask 0.6000 | best mask 0.6000", output)
        self.assertIn("M01 GPU queue : waiting for S02 (PID 20", output)
        self.assertIn("S02 CPU queue : waiting-for-training | active | PID 30", output)
        self.assertIn("M01 CPU queue : waiting-for-training | active | PID 31", output)
        self.assertIn("ONNX full-test", output)
        self.assertIn("S01-test", output)
        self.assertIn("notebook bundle : complete", output)
        self.assertIn("bundle archive  : complete", output)


if __name__ == "__main__":
    unittest.main()
