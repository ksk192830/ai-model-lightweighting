"""Tests for NVIDIA 2:4 pruning and fixed-mask recovery training."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kips_lightweighting.pruning.sparse_2to4 import (  # noqa: E402
    TwoOfFourMaskController,
    prune_two_of_four,
    verify_two_of_four,
)


class SparseTwoOfFourTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = torch.nn.Linear(8, 2, bias=False)
        with torch.no_grad():
            self.model.weight.copy_(
                torch.arange(1, 17, dtype=torch.float32).reshape(2, 8)
            )
        self.layers = [
            {
                "name": "weight",
                "type": "Linear",
                "eligible_by_shape": True,
            }
        ]

    def test_prune_and_verify(self) -> None:
        report = prune_two_of_four(
            dict(self.model.named_parameters()),
            self.layers,
        )
        self.assertTrue(report["valid"])
        self.assertEqual(report["sparsity"], 0.5)
        self.assertTrue(verify_two_of_four(self.model.weight, "Linear")["valid"])

    def test_gradient_and_post_step_masks(self) -> None:
        prune_two_of_four(dict(self.model.named_parameters()), self.layers)
        controller = TwoOfFourMaskController.from_state_dict(
            dict(self.model.named_parameters()),
            self.layers,
        )
        controller.bind_gradient_hooks(self.model)
        mask = controller.masks["weight"]

        self.model.weight.sum().backward()
        self.assertEqual(
            torch.count_nonzero(self.model.weight.grad[~mask]).item(),
            0,
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.1)
        controller.step(optimizer, self.model)
        self.assertTrue(controller.verify(self.model)["valid"])

        with torch.no_grad():
            self.model.weight[~mask] = 123
        self.assertFalse(controller.verify(self.model)["valid"])
        controller.apply(self.model)
        self.assertTrue(controller.verify(self.model)["valid"])
        controller.close()


if __name__ == "__main__":
    unittest.main()
