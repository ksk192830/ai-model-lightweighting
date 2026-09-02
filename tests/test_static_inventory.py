from __future__ import annotations

from pathlib import Path

import onnx
import torch
from onnx import TensorProto, helper

from src.kips_lightweighting.static_analysis import checkpoint_analysis, onnx_analysis


def test_checkpoint_inventory_distinguishes_state_and_prunable_elements(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.pth"
    torch.save(
        {
            "model": {
                "layer.weight": torch.tensor([[1.0, 0.0], [0.0, 2.0]]),
                "layer.bias": torch.tensor([1.0, 2.0]),
                "counter": torch.tensor(3, dtype=torch.int64),
            }
        },
        path,
    )

    report = checkpoint_analysis(path)

    assert report["model_state_tensors"] == 3
    assert report["model_state_elements"] == 7
    assert report["floating_model_state_elements"] == 6
    assert report["prunable_parameters"] == 4
    assert report["zeros"] == 2
    assert report["prunable_fraction_of_model_state_elements"] == 4 / 7
    assert report["model_state_dtype_inventory"]["float32"]["elements"] == 6


def test_onnx_initializer_inventory_reports_elements_and_storage(tmp_path: Path) -> None:
    graph = helper.make_graph(
        [helper.make_node("Add", ["input", "bias"], ["output"])],
        "initializer-inventory",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor("bias", TensorProto.FLOAT, [2], [1.0, 2.0])],
    )
    path = tmp_path / "model.onnx"
    onnx.save(helper.make_model(graph), path)

    report = onnx_analysis(path)

    assert report["onnx_initializer_elements"] == 2
    assert report["onnx_initializer_parameters"] == 2
    assert report["onnx_initializer_tensor_bytes"] == 8
    assert report["onnx_initializer_dtype_inventory"]["float"]["elements"] == 2
    assert report["excluded_operator_nodes"] == 1
    assert report["flops_status"] == "partial-lower-bound"
