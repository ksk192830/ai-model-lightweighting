import unittest

import onnx
from onnx import TensorProto, helper

from src.kips_lightweighting.static_analysis import onnx_operation_analysis


class OnnxOperationAnalysisTest(unittest.TestCase):
    def test_counts_conv_and_matmul_macs(self):
        graph = helper.make_graph(
            [
                helper.make_node("Conv", ["image", "conv_weight"], ["features"]),
                helper.make_node("MatMul", ["matrix", "linear_weight"], ["projected"]),
            ],
            "flops-test",
            [
                helper.make_tensor_value_info(
                    "image", TensorProto.FLOAT, [1, 3, 8, 8]
                ),
                helper.make_tensor_value_info(
                    "matrix", TensorProto.FLOAT, [1, 4, 8]
                ),
            ],
            [
                helper.make_tensor_value_info(
                    "features", TensorProto.FLOAT, [1, 4, 6, 6]
                ),
                helper.make_tensor_value_info(
                    "projected", TensorProto.FLOAT, [1, 4, 5]
                ),
            ],
            [
                helper.make_tensor(
                    "conv_weight",
                    TensorProto.FLOAT,
                    [4, 3, 3, 3],
                    [1.0] * (4 * 3 * 3 * 3),
                ),
                helper.make_tensor(
                    "linear_weight",
                    TensorProto.FLOAT,
                    [8, 5],
                    [1.0] * (8 * 5),
                ),
            ],
        )
        model = helper.make_model(graph)

        report = onnx_operation_analysis(model)

        expected_conv = 1 * 4 * 6 * 6 * 3 * 3 * 3
        expected_matmul = 1 * 4 * 5 * 8
        self.assertEqual(report["estimated_macs"], expected_conv + expected_matmul)
        self.assertEqual(report["estimated_flops"], 2 * (expected_conv + expected_matmul))
        self.assertEqual(report["compute_node_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
