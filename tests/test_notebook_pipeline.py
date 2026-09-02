"""Safety tests for the portable ONNX to TensorRT notebook handoff."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts" / "experiments"))

from kips_lightweighting.registry import ExperimentRegistry  # noqa: E402
from kips_lightweighting.tensorrt_build import build_command  # noqa: E402

from build_engine_suite import (  # noqa: E402
    SUITES,
    artifact_source_id,
    suite_blockers,
    suite_experiments,
)


def load_package_module():
    path = REPOSITORY_ROOT / "scripts" / "experiments" / "package_notebook_bundle.py"
    spec = importlib.util.spec_from_file_location("package_notebook_bundle", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NotebookPipelineTest(unittest.TestCase):
    def test_int8_command_uses_registered_train_split(self) -> None:
        calibration = REPOSITORY_ROOT / "data/training/front_session_split_v1/train"
        command = build_command(
            "front",
            REPOSITORY_ROOT / "shared-models/B01-front-baseline.onnx",
            REPOSITORY_ROOT / "artifacts/experiments/B03",
            "model",
            "int8",
            4096,
            dry_run=True,
            calibration_dir=calibration,
            calibration_count=128,
            calibration_seed=42,
        )
        index = command.index("--calibration-dir")
        self.assertEqual(command[index + 1], str(calibration))
        self.assertNotIn(str(calibration / "front"), command)
        self.assertEqual(command[command.index("--calibration-count") + 1], "128")
        self.assertEqual(command[command.index("--calibration-seed") + 1], "42")

    def test_suites_follow_registered_recovery_state(self) -> None:
        registry = ExperimentRegistry.load()
        self.assertEqual(suite_blockers(registry, SUITES["ready4"]), [])
        final_experiments = suite_experiments(registry, "final8")
        self.assertEqual(
            final_experiments[4],
            registry.get("C01")["artifact_source"],
        )
        blockers = suite_blockers(
            registry,
            final_experiments,
            require_artifact_onnx=False,
        )
        self.assertFalse(
            any("does not match artifact_source" in item for item in blockers)
        )
        expected_recovery_blockers = set()
        for experiment_id in final_experiments:
            experiment = registry.get(experiment_id)
            source_id = artifact_source_id(experiment_id, experiment)
            fine_tuning = registry.get(source_id).get("fine_tuning", {})
            if fine_tuning.get("required") and not fine_tuning.get("completed"):
                expected_recovery_blockers.add(
                    f"{source_id}: recovery fine-tuning is not completed"
                )
        self.assertCountEqual(blockers, expected_recovery_blockers)

    def test_shared_sources_contain_only_validated_ready_models(self) -> None:
        registry = ExperimentRegistry.load()
        sources = load_package_module().shared_onnx_sources()
        for experiment_id in SUITES["ready4"]:
            self.assertIn(experiment_id, sources)

        final_source_ids = {
            artifact_source_id(experiment_id, registry.get(experiment_id))
            for experiment_id in suite_experiments(registry, "final8")
        }
        for source_id in final_source_ids:
            fine_tuning = registry.get(source_id).get("fine_tuning", {})
            if not fine_tuning.get("required"):
                continue
            if fine_tuning.get("completed"):
                self.assertIn(source_id, sources)
            else:
                self.assertNotIn(source_id, sources)

        for experiment_id in sources:
            experiment = registry.get(experiment_id)
            source_id = artifact_source_id(experiment_id, experiment)
            fine_tuning = registry.get(source_id).get("fine_tuning", {})
            self.assertFalse(
                fine_tuning.get("required") and not fine_tuning.get("completed")
            )


if __name__ == "__main__":
    unittest.main()
