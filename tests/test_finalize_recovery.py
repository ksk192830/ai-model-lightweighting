"""Regression tests for recovery artifact promotion provenance."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_finalize_module():
    path = REPOSITORY_ROOT / "scripts" / "experiments" / "finalize_recovery.py"
    spec = importlib.util.spec_from_file_location("finalize_recovery", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


finalize = load_finalize_module()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")


class PrototypeSnapshotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name) / "S01" / "front"
        self.directory.mkdir(parents=True)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def create_prototype(self) -> None:
        (self.directory / "model.pth").write_bytes(b"prototype-pth")
        (self.directory / "model.onnx").write_bytes(b"prototype-onnx")
        write_json(
            self.directory / "metadata.json",
            {
                "artifacts": {
                    "checkpoint": str(self.directory / "model.pth"),
                    "onnx": str(self.directory / "model.onnx"),
                    "missing_build_log": str(self.directory / "build.log"),
                },
                "source_checkpoint": "/external/baseline.pth",
            },
        )
        write_json(
            self.directory / "onnx-equivalence.json",
            {
                "checkpoint": str(self.directory / "model.pth"),
                "onnx": str(self.directory / "model.onnx"),
            },
        )

    def test_snapshot_is_complete_and_rebases_json_references(self) -> None:
        self.create_prototype()

        snapshot = finalize.ensure_prototype_snapshot(self.directory)

        self.assertTrue((snapshot / "onnx-equivalence.json").is_file())
        metadata = json.loads(
            (snapshot / "metadata.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            Path(metadata["artifacts"]["checkpoint"]),
            snapshot / "model.pth",
        )
        self.assertEqual(
            metadata["source_checkpoint"],
            "/external/baseline.pth",
        )
        self.assertNotIn("missing_build_log", metadata["artifacts"])
        equivalence = json.loads(
            (snapshot / "onnx-equivalence.json").read_text(encoding="utf-8")
        )
        self.assertEqual(Path(equivalence["onnx"]), snapshot / "model.onnx")
        finalize.validate_snapshot(snapshot)
        manifest = json.loads(
            (snapshot / "prototype-snapshot.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["format_version"], 1)
        self.assertEqual(Path(manifest["snapshot_directory"]), snapshot)

    def test_repeated_snapshot_does_not_replace_original_with_recovery(self) -> None:
        self.create_prototype()
        snapshot = finalize.ensure_prototype_snapshot(self.directory)
        original = (snapshot / "model.pth").read_bytes()
        (self.directory / "model.pth").write_bytes(b"recovered-pth")
        write_json(
            self.directory / "onnx-equivalence.json",
            {"checkpoint": "recovered"},
        )

        repeated = finalize.ensure_prototype_snapshot(self.directory)

        self.assertEqual(repeated, snapshot)
        self.assertEqual((snapshot / "model.pth").read_bytes(), original)
        preserved = json.loads(
            (snapshot / "onnx-equivalence.json").read_text(encoding="utf-8")
        )
        self.assertNotEqual(preserved["checkpoint"], "recovered")

    def test_manifest_detects_snapshot_damage(self) -> None:
        self.create_prototype()
        snapshot = finalize.ensure_prototype_snapshot(self.directory)
        (snapshot / "model.pth").write_bytes(b"damaged")

        with self.assertRaisesRegex(ValueError, "Snapshot (size|hash) mismatch"):
            finalize.ensure_prototype_snapshot(self.directory)

    def test_recovery_report_is_relocated_to_snapshot(self) -> None:
        self.create_prototype()
        snapshot = finalize.ensure_prototype_snapshot(self.directory)
        source_hash = finalize.sha256(snapshot / "model.pth")
        recovery = self.directory / "recovery"
        recovered = recovery / "checkpoint_best_total.pth"
        recovered.parent.mkdir(parents=True)
        recovered.write_bytes(b"recovered-pth")
        write_json(
            recovery / "recovery-training.json",
            {
                "source_checkpoint": str(self.directory / "model.pth"),
                "source_checkpoint_sha256": source_hash,
            },
        )

        finalize.relocate_recovery_provenance(
            self.directory,
            snapshot,
            recovered,
        )

        report = json.loads(
            (recovery / "recovery-training.json").read_text(encoding="utf-8")
        )
        self.assertEqual(Path(report["source_checkpoint"]), snapshot / "model.pth")
        self.assertEqual(report["source_checkpoint_sha256"], source_hash)
        self.assertEqual(
            report["promoted_checkpoint_sha256"],
            finalize.sha256(recovered),
        )

        recovered.write_bytes(b"new-recovered-pth")
        finalize.relocate_recovery_provenance(
            self.directory,
            snapshot,
            recovered,
        )
        report = json.loads(
            (recovery / "recovery-training.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            report["promoted_checkpoint_sha256"],
            finalize.sha256(recovered),
        )

    def test_checkpoint_promotion_replaces_active_file(self) -> None:
        active = self.directory / "model.pth"
        active.write_bytes(b"prototype")
        recovered = self.directory / "recovery" / "best.pth"
        recovered.parent.mkdir()
        recovered.write_bytes(b"recovered")

        finalize.promote_checkpoint(recovered, active)

        self.assertEqual(active.read_bytes(), b"recovered")
        self.assertEqual(recovered.read_bytes(), b"recovered")
        self.assertEqual(list(self.directory.glob(".model.pth-*")), [])


if __name__ == "__main__":
    unittest.main()
