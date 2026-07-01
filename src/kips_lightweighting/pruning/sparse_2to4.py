"""NVIDIA 2:4 semi-structured magnitude pruning and verification."""

from __future__ import annotations

from typing import Any, Callable

import torch


def _groups(
    tensor: torch.Tensor,
    layer_type: str,
) -> tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    if layer_type in {"Linear", "MultiheadAttention.in_proj"}:
        if tensor.ndim != 2 or tensor.shape[1] % 4:
            raise ValueError(f"Invalid 2:4 matrix shape: {tuple(tensor.shape)}")
        grouped = tensor.reshape(tensor.shape[0], -1, 4)

        def restore(values: torch.Tensor) -> torch.Tensor:
            return values.reshape_as(tensor)

        return grouped, restore
    if layer_type == "Conv2d":
        if tensor.ndim != 4 or tensor.shape[1] % 4:
            raise ValueError(f"Invalid 2:4 convolution shape: {tuple(tensor.shape)}")
        permuted = tensor.permute(0, 2, 3, 1).contiguous()
        grouped = permuted.reshape(-1, tensor.shape[1] // 4, 4)

        def restore(values: torch.Tensor) -> torch.Tensor:
            return values.reshape_as(permuted).permute(0, 3, 1, 2).contiguous()

        return grouped, restore
    raise ValueError(f"Unsupported 2:4 layer type: {layer_type}")


def verify_two_of_four(
    tensor: torch.Tensor,
    layer_type: str,
) -> dict[str, int | float | bool]:
    grouped, _ = _groups(tensor, layer_type)
    nonzero = torch.count_nonzero(grouped, dim=-1)
    group_count = nonzero.numel()
    compliant = int((nonzero <= 2).sum().item())
    exactly_two = int((nonzero == 2).sum().item())
    return {
        "groups": group_count,
        "compliant_groups": compliant,
        "exactly_two_nonzero_groups": exactly_two,
        "compliance": compliant / group_count if group_count else 0.0,
        "valid": compliant == group_count,
    }


def prune_two_of_four(
    state: dict[str, Any],
    eligibility_layers: list[dict[str, Any]],
) -> dict[str, Any]:
    """Keep the two largest magnitudes in every eligible group of four."""
    reports = []
    missing = []
    for candidate in eligibility_layers:
        if not candidate["eligible_by_shape"]:
            continue
        name = candidate["name"]
        tensor = state.get(name)
        if not isinstance(tensor, torch.Tensor):
            missing.append(name)
            continue
        grouped, restore = _groups(tensor, candidate["type"])
        before_zeros = int(torch.count_nonzero(grouped == 0).item())
        keep = torch.topk(
            grouped.detach().abs(),
            k=2,
            dim=-1,
            largest=True,
            sorted=False,
        ).indices
        mask = torch.zeros_like(grouped, dtype=torch.bool)
        mask.scatter_(-1, keep, True)
        pruned = grouped * mask
        with torch.no_grad():
            tensor.copy_(restore(pruned))
        verification = verify_two_of_four(tensor, candidate["type"])
        reports.append(
            {
                "name": name,
                "type": candidate["type"],
                "shape": list(tensor.shape),
                "parameters": tensor.numel(),
                "zeros_before": before_zeros,
                "zeros_after": int(torch.count_nonzero(tensor == 0).item()),
                **verification,
            }
        )
    if missing:
        raise ValueError(
            f"Eligible weights missing from checkpoint ({len(missing)}): "
            + ", ".join(missing[:5])
        )
    groups = sum(report["groups"] for report in reports)
    compliant = sum(report["compliant_groups"] for report in reports)
    parameters = sum(report["parameters"] for report in reports)
    zeros = sum(report["zeros_after"] for report in reports)
    return {
        "pattern": "2:4",
        "layers": len(reports),
        "parameters": parameters,
        "zeros": zeros,
        "sparsity": zeros / parameters if parameters else 0.0,
        "groups": groups,
        "compliant_groups": compliant,
        "compliance": compliant / groups if groups else 0.0,
        "valid": compliant == groups,
        "layer_reports": reports,
    }


class TwoOfFourMaskController:
    """Keep an existing 2:4 mask fixed during recovery fine-tuning.

    Gradient hooks prevent updates on pruned entries. ``apply`` must additionally
    run after every optimizer step because momentum or optimizer state can
    otherwise regrow a zero weight.
    """

    def __init__(
        self,
        masks: dict[str, torch.Tensor],
        layer_types: dict[str, str],
    ) -> None:
        self.masks = {
            name: mask.detach().to(device="cpu", dtype=torch.bool).clone()
            for name, mask in masks.items()
        }
        self.layer_types = dict(layer_types)
        self._handles: list[Any] = []

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, Any],
        eligibility_layers: list[dict[str, Any]],
    ) -> "TwoOfFourMaskController":
        masks = {}
        layer_types = {}
        missing = []
        for layer in eligibility_layers:
            if not layer["eligible_by_shape"]:
                continue
            name = layer["name"]
            tensor = state.get(name)
            if not isinstance(tensor, torch.Tensor):
                missing.append(name)
                continue
            verification = verify_two_of_four(tensor, layer["type"])
            if not verification["valid"]:
                raise ValueError(f"Weight does not satisfy 2:4 pattern: {name}")
            masks[name] = tensor.detach().ne(0)
            layer_types[name] = layer["type"]
        if missing:
            raise ValueError(
                f"Mask weights missing from state dict ({len(missing)}): "
                + ", ".join(missing[:5])
            )
        return cls(masks, layer_types)

    def _parameters(self, model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
        parameters = dict(model.named_parameters())
        missing = sorted(set(self.masks).difference(parameters))
        if missing:
            raise ValueError(
                f"Mask parameters missing from model ({len(missing)}): "
                + ", ".join(missing[:5])
            )
        for name, mask in self.masks.items():
            if parameters[name].shape != mask.shape:
                raise ValueError(
                    f"Mask shape mismatch for {name}: "
                    f"{tuple(mask.shape)} != {tuple(parameters[name].shape)}"
                )
        return parameters

    def bind_gradient_hooks(self, model: torch.nn.Module) -> None:
        """Register hooks that zero gradients at every pruned position."""
        self.close()
        parameters = self._parameters(model)
        for name, mask in self.masks.items():
            parameter = parameters[name]

            def mask_gradient(
                gradient: torch.Tensor,
                fixed_mask: torch.Tensor = mask,
            ) -> torch.Tensor:
                return gradient * fixed_mask.to(gradient.device)

            self._handles.append(parameter.register_hook(mask_gradient))

    def mask_gradients(self, model: torch.nn.Module) -> None:
        """Explicit gradient masking used as a callback-level safety check."""
        parameters = self._parameters(model)
        for name, mask in self.masks.items():
            parameter = parameters[name]
            if parameter.grad is None:
                continue
            parameter.grad.mul_(mask.to(parameter.grad.device))

    @torch.no_grad()
    def apply(self, model: torch.nn.Module) -> None:
        """Reapply masks after an optimizer step."""
        parameters = self._parameters(model)
        for name, mask in self.masks.items():
            parameter = parameters[name]
            parameter.mul_(mask.to(parameter.device))

    def step(
        self,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        closure: Callable[[], Any] | None = None,
    ) -> Any:
        """Mask gradients, execute one optimizer step, then restore all zeros."""
        self.mask_gradients(model)
        result = optimizer.step(closure=closure) if closure else optimizer.step()
        self.apply(model)
        return result

    def verify(self, model: torch.nn.Module) -> dict[str, Any]:
        parameters = self._parameters(model)
        invalid = []
        regrown = 0
        for name, mask in self.masks.items():
            tensor = parameters[name].detach()
            regrown_count = int(torch.count_nonzero(tensor[~mask]).item())
            verification = verify_two_of_four(tensor, self.layer_types[name])
            if regrown_count or not verification["valid"]:
                invalid.append(name)
                regrown += regrown_count
        return {
            "layers": len(self.masks),
            "valid_layers": len(self.masks) - len(invalid),
            "invalid_layers": invalid,
            "regrown_weights": regrown,
            "valid": not invalid,
        }

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def make_lightning_mask_callback(
    controller: TwoOfFourMaskController,
):
    """Create a PyTorch Lightning callback without a hard package dependency."""
    try:
        from pytorch_lightning import Callback
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "PyTorch Lightning is required for RF-DETR recovery fine-tuning."
        ) from error

    class TwoOfFourMaskCallback(Callback):
        def setup(self, trainer, pl_module, stage: str) -> None:
            if stage == "fit":
                controller.bind_gradient_hooks(pl_module.model)
                controller.apply(pl_module.model)

        def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
            controller.mask_gradients(pl_module.model)

        def on_before_zero_grad(self, trainer, pl_module, optimizer) -> None:
            # Lightning calls this after optimizer.step and before zero_grad.
            controller.apply(pl_module.model)

        def on_train_batch_end(
            self,
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx: int,
        ) -> None:
            # Also covers optimizer/accumulation strategy differences.
            controller.apply(pl_module.model)

        def teardown(self, trainer, pl_module, stage: str) -> None:
            controller.close()

    return TwoOfFourMaskCallback()
