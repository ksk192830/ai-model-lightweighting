"""Static checkpoint and ONNX analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def is_prunable(name: str, tensor: Any) -> bool:
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.is_floating_point()
        and tensor.ndim >= 2
        and name.endswith("weight")
    )


def checkpoint_state(checkpoint: dict[str, Any]) -> dict[str, Any]:
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a 'model' state dict.")
    return state


def checkpoint_analysis(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint_state(checkpoint)
    layers = []
    total = 0
    zeros = 0
    for name, tensor in state.items():
        if not is_prunable(name, tensor):
            continue
        count = tensor.numel()
        zero_count = int(torch.count_nonzero(tensor == 0).item())
        total += count
        zeros += zero_count
        layers.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "parameters": count,
                "zeros": zero_count,
                "sparsity": zero_count / count if count else 0.0,
            }
        )
    return {
        "checkpoint_size_bytes": path.stat().st_size,
        "prunable_layers": len(layers),
        "prunable_parameters": total,
        "zeros": zeros,
        "sparsity": zeros / total if total else 0.0,
        "layers": layers,
    }


def onnx_analysis(path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(path))
    onnx.checker.check_model(model)

    def value_shape(value) -> list[int | str]:
        return [
            dimension.dim_value
            if dimension.dim_value
            else dimension.dim_param or "?"
            for dimension in value.type.tensor_type.shape.dim
        ]

    initializers = sum(
        int(torch.tensor(initializer.dims).prod().item())
        for initializer in model.graph.initializer
        if initializer.dims
    )
    return {
        "onnx_size_bytes": path.stat().st_size,
        "onnx_nodes": len(model.graph.node),
        "onnx_initializers": len(model.graph.initializer),
        "onnx_initializer_parameters": initializers,
        "estimated_flops": None,
        "flops_status": "pending operator-aware shape analysis",
        "onnx_checker_valid": True,
        "inputs": {
            value.name: value_shape(value) for value in model.graph.input
        },
        "outputs": {
            value.name: value_shape(value) for value in model.graph.output
        },
    }


def two_to_four_eligibility(model: torch.nn.Module) -> dict[str, Any]:
    """Inventory weights that can theoretically satisfy TensorRT 2:4 layout."""
    import torch.nn as nn

    layers = []
    seen: set[str] = set()

    def add(
        name: str,
        layer_type: str,
        weight: torch.Tensor,
        reduction_axis_size: int,
        eligible: bool,
        reason: str,
    ) -> None:
        seen.add(name)
        groups = weight.numel() // 4 if eligible else 0
        compliant_groups = 0
        if eligible:
            if weight.ndim == 2:
                grouped = weight.reshape(weight.shape[0], -1, 4)
            else:
                grouped = (
                    weight.permute(0, 2, 3, 1)
                    .contiguous()
                    .reshape(-1, reduction_axis_size // 4, 4)
                )
            compliant_groups = int(
                (torch.count_nonzero(grouped, dim=-1) <= 2).sum().item()
            )
            groups = grouped.shape[0] * grouped.shape[1]
        layers.append(
            {
                "name": name,
                "type": layer_type,
                "shape": list(weight.shape),
                "parameters": weight.numel(),
                "reduction_axis_size": reduction_axis_size,
                "eligible_by_shape": eligible,
                "reason": reason,
                "pattern_groups": groups,
                "compliant_groups_before_pruning": compliant_groups,
                "pattern_compliance_before_pruning": (
                    compliant_groups / groups if groups else None
                ),
            }
        )

    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            name = f"{module_name}.weight"
            eligible = module.in_features % 4 == 0
            add(
                name,
                "Linear",
                module.weight.detach(),
                module.in_features,
                eligible,
                "input features divisible by 4"
                if eligible
                else "input features not divisible by 4",
            )
        elif isinstance(module, nn.Conv2d):
            name = f"{module_name}.weight"
            channels_per_group = module.in_channels // module.groups
            eligible = channels_per_group % 4 == 0
            add(
                name,
                "Conv2d",
                module.weight.detach(),
                channels_per_group,
                eligible,
                "input channels per group divisible by 4"
                if eligible
                else "input channels per group not divisible by 4",
            )
        elif isinstance(module, nn.MultiheadAttention):
            if module.in_proj_weight is not None:
                name = f"{module_name}.in_proj_weight"
                eligible = module.embed_dim % 4 == 0
                add(
                    name,
                    "MultiheadAttention.in_proj",
                    module.in_proj_weight.detach(),
                    module.embed_dim,
                    eligible,
                    "embedding dimension divisible by 4"
                    if eligible
                    else "embedding dimension not divisible by 4",
                )

    eligible_layers = [layer for layer in layers if layer["eligible_by_shape"]]
    return {
        "total_layers": len(layers),
        "eligible_layers_by_shape": len(eligible_layers),
        "ineligible_layers_by_shape": len(layers) - len(eligible_layers),
        "total_parameters": sum(layer["parameters"] for layer in layers),
        "eligible_parameters_by_shape": sum(
            layer["parameters"] for layer in eligible_layers
        ),
        "notes": [
            "Shape eligibility does not guarantee TensorRT sparse tactic selection.",
            "ONNX must contain a constant-weight Convolution or MatrixMultiply.",
            "TensorRT verbose build logs are the final source of tactic selection.",
        ],
        "layers": layers,
    }


def onnx_two_to_four_analysis(path: Path) -> dict[str, Any]:
    """Verify 2:4 compliance on constant Conv/MatMul/Gemm ONNX weights."""
    import numpy as np
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path))
    constants = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    layers = []

    def verify(name: str, op_type: str, weight, axis: int) -> None:
        size = weight.shape[axis]
        eligible = size % 4 == 0
        groups = compliant = 0
        if eligible:
            arranged = np.moveaxis(weight, axis, -1)
            grouped = arranged.reshape(-1, size // 4, 4)
            nonzero = np.count_nonzero(grouped, axis=-1)
            groups = int(nonzero.size)
            compliant = int(np.count_nonzero(nonzero <= 2))
        layers.append(
            {
                "name": name,
                "op_type": op_type,
                "shape": list(weight.shape),
                "reduction_axis": axis,
                "eligible_by_shape": eligible,
                "groups": groups,
                "compliant_groups": compliant,
                "compliance": compliant / groups if groups else None,
                "valid": eligible and compliant == groups,
            }
        )

    for node in model.graph.node:
        if node.op_type == "Conv" and len(node.input) > 1:
            weight = constants.get(node.input[1])
            if weight is not None and weight.ndim == 4:
                verify(node.name or node.output[0], "Conv", weight, 1)
        elif node.op_type == "MatMul" and len(node.input) == 2:
            left = constants.get(node.input[0])
            right = constants.get(node.input[1])
            if right is not None and right.ndim >= 2:
                verify(node.name or node.output[0], "MatMul", right, -2)
            elif left is not None and left.ndim >= 2:
                verify(node.name or node.output[0], "MatMul", left, -1)
        elif node.op_type == "Gemm" and len(node.input) > 1:
            weight = constants.get(node.input[1])
            if weight is None or weight.ndim != 2:
                continue
            attributes = {attribute.name: attribute.i for attribute in node.attribute}
            verify(
                node.name or node.output[0],
                "Gemm",
                weight,
                1 if attributes.get("transB", 0) else 0,
            )

    eligible = [layer for layer in layers if layer["eligible_by_shape"]]
    compliant = [layer for layer in eligible if layer["valid"]]
    return {
        "onnx_path": str(path),
        "constant_weight_ops": len(layers),
        "eligible_ops_by_shape": len(eligible),
        "compliant_ops": len(compliant),
        "compliance": len(compliant) / len(eligible) if eligible else 0.0,
        "valid": len(compliant) == len(eligible) and bool(eligible),
        "layers": layers,
    }
