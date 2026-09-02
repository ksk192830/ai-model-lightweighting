"""Static safety checks for the self-contained ready4 Colab notebook."""

from __future__ import annotations

import hashlib
import json
import tarfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPOSITORY_ROOT / "notebooks" / "front_tensorrt_ready4.ipynb"
ARCHIVE_PATH = REPOSITORY_ROOT / "delivery" / "notebook-front-ready4.tar.gz"
PAYLOAD_ROOT = REPOSITORY_ROOT / "delivery" / "notebook-front-ready4"
EXPECTED_ARCHIVE_SHA256 = (
    "9b4a9ed3bb1e097351e143f6e6d03031ef15f802e055cd17d2637ffdc827a722"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FrontTensorRTNotebookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.code = [
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "code"
        ]
        cls.all_code = "\n".join(cls.code)
        cls.manifest = json.loads(
            (PAYLOAD_ROOT / "manifest.json").read_text(encoding="utf-8")
        )

    def test_notebook_json_and_every_code_cell_compile(self) -> None:
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertEqual(self.notebook["metadata"]["accelerator"], "GPU")
        for index, source in enumerate(self.code):
            with self.subTest(code_cell=index):
                compile(source, f"{NOTEBOOK_PATH.name}:cell-{index}", "exec")

    def test_archive_and_manifest_are_the_pinned_ready4_payload(self) -> None:
        self.assertEqual(sha256(ARCHIVE_PATH), EXPECTED_ARCHIVE_SHA256)
        self.assertIn(EXPECTED_ARCHIVE_SHA256, self.all_code)
        plan = sorted(self.manifest["engine_plan"], key=lambda item: item["order"])
        self.assertEqual(
            [item["experiment_id"] for item in plan],
            ["B01", "B02", "B03", "R01"],
        )
        self.assertEqual(
            self.manifest["calibration_dir"],
            "data/training/front_session_split_v1/train",
        )
        for source in self.manifest["onnx_sources"]:
            path = PAYLOAD_ROOT / source["path"]
            self.assertTrue(path.is_file())
            self.assertEqual(path.stat().st_size, source["size_bytes"])
            self.assertEqual(sha256(path), source["sha256"])

    def test_archive_contains_only_safe_regular_files_and_directories(self) -> None:
        with tarfile.open(ARCHIVE_PATH, "r:gz") as archive:
            for member in archive.getmembers():
                path = Path(member.name)
                self.assertFalse(path.is_absolute())
                self.assertNotIn("..", path.parts)
                self.assertEqual(path.parts[0], "notebook-front-ready4")
                self.assertTrue(member.isfile() or member.isdir())
                self.assertFalse(member.issym() or member.islnk() or member.isdev())

    def test_dry_run_and_sequential_build_cells_are_present(self) -> None:
        dry_run_cell = next(source for source in self.code if "target_engine" in source)
        self.assertIn("calibration", dry_run_cell)
        call_cells = [
            source.strip()
            for source in self.code
            if source.strip().startswith(("B01_ENGINE", "B02_ENGINE", "B03_ENGINE", "R01_ENGINE"))
        ]
        self.assertEqual(
            call_cells,
            [
                'B01_ENGINE = build_engine("B01")  # 1/4: baseline FP32',
                'B02_ENGINE = build_engine("B02")  # 2/4: baseline FP16',
                'B03_ENGINE = build_engine("B03")  # 3/4: train 128장 INT8 PTQ',
                'R01_ENGINE = build_engine("R01")  # 4/4: 432x432 FP16',
            ],
        )

    def test_calibration_is_train_only_and_test_is_437_image_evaluation(self) -> None:
        data_manifest = json.loads(
            (PAYLOAD_ROOT / "data-manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(data_manifest["calibration"]["source_split"], "train")
        self.assertEqual(data_manifest["calibration"]["selected_images"], 128)
        self.assertEqual(data_manifest["test"]["summary"]["images"], 437)
        self.assertIn("len(images) != 128", self.all_code)
        self.assertIn("len(test_images) != 437", self.all_code)
        self.assertNotIn("PostProcess(num_select=300)", self.all_code)
        self.assertIn(
            "PostProcess(num_select=self.postprocess_num_select)", self.all_code
        )
        self.assertIn(
            "runner.postprocess_num_select != runner.query_count", self.all_code
        )


if __name__ == "__main__":
    unittest.main()
