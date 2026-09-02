"""Regression tests for terminal decisions in artifact auditing."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from kips_lightweighting.artifacts import artifact_paths  # noqa: E402
from kips_lightweighting.metadata import sha256  # noqa: E402


def load_audit_module():
    path = REPOSITORY_ROOT / "scripts" / "experiments" / "audit_artifacts.py"
    spec = importlib.util.spec_from_file_location("audit_artifacts", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeRegistry:
    schema = {"metadata_required_fields": []}

    def __init__(self, experiment: dict):
        self.experiments = {"S03": experiment}

    def get(self, experiment_id: str) -> dict:
        return self.experiments[experiment_id]


class RejectedArtifactAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.artifact_root = self.root / "artifacts" / "experiments"
        self.paths = artifact_paths("S03", "front", self.artifact_root)
        self.paths.directory.mkdir(parents=True)
        self.paths.checkpoint.write_bytes(b"checkpoint")
        self.paths.onnx.write_bytes(b"onnx")
        self.paths.static_analysis.write_text("{}", encoding="utf-8")
        self.equivalence = self.paths.directory / "onnx-equivalence.json"
        self.equivalence.write_text(
            json.dumps({"passed": False}),
            encoding="utf-8",
        )
        self.experiment = {
            "family": "structured",
            "cameras": ["front"],
            "status": "static-analysis-rejected",
            "result": {
                "decision": "rejected",
                "reason": "ONNX equivalence threshold failed.",
            },
        }
        self.write_metadata()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_metadata(self, artifact_hash: str | None = None) -> None:
        metadata = {
            "experiment_id": "S03",
            "camera": "front",
            "status": "static-analysis-rejected",
            "source_checkpoint": str(self.paths.checkpoint),
            "source_checkpoint_sha256": sha256(self.paths.checkpoint),
            "artifact_checkpoint_sha256": (
                artifact_hash or sha256(self.paths.checkpoint)
            ),
            "artifacts": {
                "checkpoint": str(self.paths.checkpoint),
                "onnx": str(self.paths.onnx),
                "onnx_equivalence": str(self.equivalence),
            },
        }
        self.paths.metadata.write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )

    def run_audit(self) -> tuple[int, dict]:
        module = load_audit_module()
        output = self.root / "audit.json"
        registry = FakeRegistry(self.experiment)
        with (
            mock.patch.object(module, "REPOSITORY_ROOT", self.root),
            mock.patch.object(module.ExperimentRegistry, "load", return_value=registry),
            mock.patch.object(
                module,
                "artifact_paths",
                side_effect=lambda experiment_id, camera: artifact_paths(
                    experiment_id,
                    camera,
                    self.artifact_root,
                ),
            ),
            mock.patch.object(
                sys,
                "argv",
                ["audit_artifacts.py", "S03", "--output", str(output)],
            ),
        ):
            return_code = module.main()
        return return_code, json.loads(output.read_text(encoding="utf-8"))

    def test_rejected_status_survives_onnx_and_failed_equivalence(self) -> None:
        return_code, report = self.run_audit()

        result = report["results"][0]
        self.assertEqual(return_code, 0)
        self.assertTrue(result["valid"])
        self.assertEqual(
            result["resolved_artifact_status"],
            "static-analysis-rejected",
        )

    def test_missing_referenced_artifact_is_still_an_error(self) -> None:
        self.paths.onnx.unlink()

        return_code, report = self.run_audit()

        codes = {issue["code"] for issue in report["results"][0]["issues"]}
        self.assertEqual(return_code, 1)
        self.assertIn("broken-artifact-reference", codes)

    def test_checkpoint_hash_mismatch_is_still_an_error(self) -> None:
        self.write_metadata(artifact_hash="incorrect")

        return_code, report = self.run_audit()

        codes = {issue["code"] for issue in report["results"][0]["issues"]}
        self.assertEqual(return_code, 1)
        self.assertIn("artifact-checkpoint-hash", codes)


if __name__ == "__main__":
    unittest.main()
