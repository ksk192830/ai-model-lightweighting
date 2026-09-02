"""Safety checks for the CPU-only post-recovery queue."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_queue_module():
    path = REPOSITORY_ROOT / "scripts/experiments/run_post_recovery_queue.py"
    spec = importlib.util.spec_from_file_location("run_post_recovery_queue", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


queue = load_queue_module()


class CompletedTrainingValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / "checkpoint_best_total.pth"
        self.checkpoint.write_bytes(b"recovered")
        self.report = self.root / "recovery-training.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_report(self, **updates) -> None:
        data = {
            "experiment_id": "S02",
            "camera": "front",
            "status": "completed",
            "structure_verification": {"valid": True},
            "best_checkpoint": str(self.checkpoint),
            "best_checkpoint_sha256": queue.sha256(self.checkpoint),
        }
        data.update(updates)
        self.report.write_text(json.dumps(data) + "\n", encoding="utf-8")

    def test_accepts_completed_verified_checkpoint(self) -> None:
        self.write_report()

        _, checkpoint = queue.validate_completed_training(
            self.report, "S02", "front"
        )

        self.assertEqual(checkpoint, self.checkpoint)

    def test_rejects_running_training_before_promotion(self) -> None:
        self.write_report(status="running")

        with self.assertRaisesRegex(ValueError, "not completed"):
            queue.validate_completed_training(self.report, "S02", "front")

    def test_rejects_checkpoint_hash_mismatch(self) -> None:
        self.write_report(best_checkpoint_sha256="incorrect")

        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            queue.validate_completed_training(self.report, "S02", "front")

    def test_rejects_failed_structure_verification(self) -> None:
        self.write_report(structure_verification={"valid": False})

        with self.assertRaisesRegex(ValueError, "structure verification"):
            queue.validate_completed_training(self.report, "S02", "front")


if __name__ == "__main__":
    unittest.main()
