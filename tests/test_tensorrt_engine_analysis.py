from __future__ import annotations

import importlib.util
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = REPOSITORY_ROOT / "scripts/experiments/analyze_tensorrt_engine.py"
    spec = importlib.util.spec_from_file_location("analyze_tensorrt_engine", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_precision_evidence_is_non_exclusive() -> None:
    module = load_module()
    layer = {
        "Inputs": [{"Format/Datatype": "Float"}],
        "Outputs": [{"Format/Datatype": "Half"}],
        "TacticName": "int8_tensor_core",
    }
    assert module.precision_evidence(layer) == {"fp32", "fp16", "int8"}


def test_inspector_json_accepts_list_and_layer_mapping() -> None:
    module = load_module()
    layers, error = module.parse_inspector_json('[{"Name": "one"}]')
    assert error is None
    assert layers == [{"Name": "one"}]
    layers, error = module.parse_inspector_json('{"Layers": [{"Name": "two"}]}')
    assert error is None
    assert layers == [{"Name": "two"}]
