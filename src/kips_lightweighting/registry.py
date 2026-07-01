"""Load and validate the experiment registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULTS_PATH = REPOSITORY_ROOT / "configs" / "experiments" / "defaults.yaml"
REGISTRY_PATH = REPOSITORY_ROOT / "configs" / "experiments" / "registry.yaml"
SCHEMA_PATH = REPOSITORY_ROOT / "configs" / "experiments" / "schema.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return data


@dataclass(frozen=True)
class ExperimentRegistry:
    defaults: dict[str, Any]
    experiments: dict[str, dict[str, Any]]
    schema: dict[str, Any]

    @classmethod
    def load(
        cls,
        registry_path: Path = REGISTRY_PATH,
        defaults_path: Path = DEFAULTS_PATH,
        schema_path: Path = SCHEMA_PATH,
    ) -> "ExperimentRegistry":
        registry_data = _load_yaml(registry_path)
        registry = cls(
            defaults=_load_yaml(defaults_path),
            experiments=registry_data.get("experiments", {}),
            schema=_load_yaml(schema_path),
        )
        registry.validate()
        return registry

    def validate(self) -> None:
        if not isinstance(self.experiments, dict) or not self.experiments:
            raise ValueError("Experiment registry is empty.")
        required = set(self.schema["required_experiment_fields"])
        allowed_cameras = set(self.schema["allowed_cameras"])
        allowed_precisions = set(self.schema["allowed_precisions"])
        allowed_statuses = set(self.schema["allowed_statuses"])
        for experiment_id, experiment in self.experiments.items():
            if not isinstance(experiment_id, str) or not experiment_id.isalnum():
                raise ValueError(f"Invalid experiment ID: {experiment_id!r}")
            missing = required.difference(experiment)
            if missing:
                raise ValueError(
                    f"{experiment_id} is missing fields: {sorted(missing)}"
                )
            cameras = experiment["cameras"]
            if not cameras or not set(cameras).issubset(allowed_cameras):
                raise ValueError(f"{experiment_id} has invalid cameras: {cameras}")
            if experiment["precision"] not in allowed_precisions:
                raise ValueError(
                    f"{experiment_id} has invalid precision: "
                    f"{experiment['precision']}"
                )
            if experiment["status"] not in allowed_statuses:
                raise ValueError(
                    f"{experiment_id} has invalid status: {experiment['status']}"
                )
            unknown_dependencies = set(
                experiment.get("depends_on", [])
            ).difference(self.experiments)
            if unknown_dependencies:
                raise ValueError(
                    f"{experiment_id} has unknown dependencies: "
                    f"{sorted(unknown_dependencies)}"
                )

    def get(self, experiment_id: str) -> dict[str, Any]:
        try:
            return self.experiments[experiment_id]
        except KeyError as error:
            raise KeyError(f"Unknown experiment ID: {experiment_id}") from error

    def cameras(self, experiment_id: str) -> list[str]:
        return list(self.get(experiment_id)["cameras"])
