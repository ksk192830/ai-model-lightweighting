import unittest

import torch

from src.kips_lightweighting.rfdetr_compat import (
    _DECODER_LAYER,
    _contiguous_count,
)
from src.kips_lightweighting.pruning.structured import (
    prune_decoder_layers,
    prune_ffn_dimensions,
)


class StructuredCheckpointDetectionTest(unittest.TestCase):
    def test_counts_contiguous_decoder_layers(self):
        state = {
            f"transformer.decoder.layers.{index}.linear1.weight": torch.ones(1)
            for index in range(3)
        }
        self.assertEqual(_contiguous_count(state, _DECODER_LAYER), 3)

    def test_rejects_noncontiguous_decoder_layers(self):
        state = {
            "transformer.decoder.layers.0.linear1.weight": torch.ones(1),
            "transformer.decoder.layers.2.linear1.weight": torch.ones(1),
        }
        with self.assertRaises(ValueError):
            _contiguous_count(state, _DECODER_LAYER)


class StructuredDecoderPruningTest(unittest.TestCase):
    def test_removes_trailing_decoder_and_segmentation_layers(self):
        state = {}
        for index in range(3):
            state[f"transformer.decoder.layers.{index}.linear.weight"] = (
                torch.ones(2, 2)
            )
            state[f"segmentation_head.blocks.{index}.conv.weight"] = (
                torch.ones(2, 2)
            )
        state["backbone.weight"] = torch.ones(2, 2)
        checkpoint = {
            "model": state,
            "state_dict": {
                f"model.{name}": value.clone() for name, value in state.items()
            },
            "args": {},
        }

        report = prune_decoder_layers(
            checkpoint,
            remove_layers=1,
            model_config={"dec_layers": 3},
        )

        self.assertEqual(report["remaining_decoder_layers"], 2)
        self.assertEqual(report["removed_decoder_layers"], [2])
        self.assertNotIn(
            "transformer.decoder.layers.2.linear.weight",
            checkpoint["model"],
        )
        self.assertNotIn(
            "model.segmentation_head.blocks.2.conv.weight",
            checkpoint["state_dict"],
        )
        self.assertEqual(checkpoint["model_config"]["dec_layers"], 2)
        self.assertEqual(checkpoint["args"]["dec_layers"], 2)

    def test_reduces_ffn_hidden_dimension_as_a_structural_group(self):
        state = {}
        for index in range(2):
            state[f"transformer.decoder.layers.{index}.linear1.weight"] = (
                torch.arange(64 * 8, dtype=torch.float32).reshape(64, 8)
            )
            state[f"transformer.decoder.layers.{index}.linear1.bias"] = (
                torch.arange(64, dtype=torch.float32)
            )
            state[f"transformer.decoder.layers.{index}.linear2.weight"] = (
                torch.arange(8 * 64, dtype=torch.float32).reshape(8, 64)
            )
            state[f"transformer.decoder.layers.{index}.linear2.bias"] = (
                torch.ones(8)
            )
        checkpoint = {"model": state}

        report = prune_ffn_dimensions(
            checkpoint,
            reduction=0.25,
            alignment=16,
        )

        self.assertEqual(report["remaining_dim_feedforward"], 48)
        self.assertEqual(
            checkpoint["model"][
                "transformer.decoder.layers.0.linear1.weight"
            ].shape,
            (48, 8),
        )
        self.assertEqual(
            checkpoint["model"][
                "transformer.decoder.layers.0.linear2.weight"
            ].shape,
            (8, 48),
        )
        self.assertEqual(
            checkpoint["structured_architecture"]["dim_feedforward"],
            48,
        )


if __name__ == "__main__":
    unittest.main()
