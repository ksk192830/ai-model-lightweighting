#!/usr/bin/env python3
"""Evaluate detection models on a fixed validation/test split.

Supported backends:
- ultralytics: YOLO-style models supported by the Ultralytics package.
- rfdetr: RF-DETR checkpoints when the optional rfdetr package is installed.
- rfdetr_engine: RF-DETR TensorRT .engine files exported from RF-DETR ONNX.

The evaluator intentionally computes detection metrics locally so the same
thresholds and split handling can be reused by benchmark.py.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import io
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import yaml
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
IOU_THRESHOLDS = tuple(round(x, 2) for x in np.arange(0.50, 0.96, 0.05))


@dataclass(frozen=True)
class GroundTruth:
    image_id: str
    class_id: int
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class Prediction:
    image_id: str
    class_id: int
    confidence: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class DatasetSample:
    image_id: str
    image_path: Path
    width: int
    height: int
    targets: tuple[GroundTruth, ...]


class EvaluationError(RuntimeError):
    """Raised when evaluation cannot run with the supplied inputs."""


class BasePredictor:
    backend_name = "base"

    def predict_batch(self, image_paths: Sequence[Path]) -> list[list[Prediction]]:
        raise NotImplementedError

    def close(self) -> None:
        return None


class UltralyticsPredictor(BasePredictor):
    backend_name = "ultralytics"

    def __init__(
        self,
        model_path: Path,
        imgsz: int,
        conf: float,
        iou: float,
        device: str,
        batch_size: int,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise EvaluationError("ultralytics is not installed. Install it or choose another backend.") from exc

        self.model = YOLO(str(model_path))
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = normalize_device_for_ultralytics(device)
        self.batch_size = batch_size

    def predict_batch(self, image_paths: Sequence[Path]) -> list[list[Prediction]]:
        results = self.model.predict(
            source=[str(path) for path in image_paths],
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            batch=self.batch_size,
            verbose=False,
        )

        predictions: list[list[Prediction]] = []
        for image_path, result in zip(image_paths, results):
            per_image: list[Prediction] = []
            boxes = getattr(result, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                confs = boxes.conf.detach().cpu().numpy()
                classes = boxes.cls.detach().cpu().numpy().astype(int)
                for box, score, class_id in zip(xyxy, confs, classes):
                    per_image.append(
                        Prediction(
                            image_id=image_id_for_path(image_path),
                            class_id=int(class_id),
                            confidence=float(score),
                            box=tuple(float(v) for v in box),
                        )
                    )
            predictions.append(per_image)
        return predictions


class RFDETRPredictor(BasePredictor):
    backend_name = "rfdetr"

    def __init__(
        self,
        model_path: Path,
        imgsz: int,
        conf: float,
        iou: float,
        device: str,
        batch_size: int,
    ) -> None:
        del iou, batch_size
        try:
            RFDETR = importlib.import_module("rfdetr.detr").RFDETR
        except ImportError as exc:
            raise EvaluationError(
                "rfdetr is not installed. Install with `pip install rfdetr` to evaluate RF-DETR .pth checkpoints."
            ) from exc

        self.model = RFDETR.from_checkpoint(str(model_path))
        self.imgsz = imgsz
        self.conf = conf
        self.device = resolve_torch_device(device)
        self._move_model_to_device()

    def _move_model_to_device(self) -> None:
        context = getattr(self.model, "model", None)
        module = getattr(context, "model", None)
        if module is not None and hasattr(module, "to"):
            module.to(self.device)
        if context is not None and hasattr(context, "device"):
            context.device = self.device

    def predict_batch(self, image_paths: Sequence[Path]) -> list[list[Prediction]]:
        raw = self.model.predict(
            [str(path) for path in image_paths],
            threshold=self.conf,
            shape=(self.imgsz, self.imgsz),
            include_source_image=False,
        )
        raw_results = raw if isinstance(raw, list) else [raw]

        predictions: list[list[Prediction]] = []
        for image_path, result in zip(image_paths, raw_results):
            xyxy = np.asarray(getattr(result, "xyxy", []), dtype=float)
            confs = np.asarray(getattr(result, "confidence", []), dtype=float)
            classes = np.asarray(getattr(result, "class_id", []), dtype=int)
            per_image = [
                Prediction(
                    image_id=image_id_for_path(image_path),
                    class_id=int(class_id),
                    confidence=float(score),
                    box=tuple(float(v) for v in box),
                )
                for box, score, class_id in zip(xyxy, confs, classes)
            ]
            predictions.append(per_image)
        return predictions


class RFDETRTensorRTPredictor(BasePredictor):
    backend_name = "rfdetr_engine"

    def __init__(
        self,
        model_path: Path,
        imgsz: int,
        conf: float,
        iou: float,
        device: str,
        batch_size: int,
    ) -> None:
        del iou
        if not torch.cuda.is_available():
            raise EvaluationError("TensorRT .engine inference requires CUDA, but CUDA is not available.")
        self.trt = import_tensorrt()
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf = conf
        self.device = resolve_torch_device(device)
        if self.device.type != "cuda":
            raise EvaluationError("TensorRT .engine inference requires a CUDA device. Use --device cuda:0.")
        self.requested_batch_size = batch_size
        self.logger = self.trt.Logger(self.trt.Logger.WARNING)
        self.trt.init_libnvinfer_plugins(self.logger, "")
        self.engine = self._load_engine(model_path)
        self.context = self.engine.create_execution_context()
        self.tensor_names = engine_tensor_names(self.engine)
        self.input_names = [name for name in self.tensor_names if tensor_is_input(self.engine, self.trt, name)]
        self.output_names = [name for name in self.tensor_names if not tensor_is_input(self.engine, self.trt, name)]
        if len(self.input_names) != 1:
            raise EvaluationError(f"Expected exactly one TensorRT input tensor, found {self.input_names}")
        if not self.output_names:
            raise EvaluationError("TensorRT engine has no output tensors.")
        self.input_name = self.input_names[0]
        self.input_dtype = np.dtype(self.trt.nptype(get_tensor_dtype(self.engine, self.input_name)))
        if self.input_dtype not in {np.dtype("float32"), np.dtype("float16")}:
            raise EvaluationError(
                f"Unsupported TensorRT input dtype {self.input_dtype}. Expected float32 or float16 preprocessed input."
            )
        self.static_batch_size, self.channels, self.input_height, self.input_width = self._resolve_input_shape()
        self.stream = torch.cuda.Stream(device=self.device)

    def _load_engine(self, path: Path) -> Any:
        with path.open("rb") as handle, self.trt.Runtime(self.logger) as runtime:
            engine = runtime.deserialize_cuda_engine(handle.read())
        if engine is None:
            raise EvaluationError(f"Failed to deserialize TensorRT engine: {path}")
        return engine

    def _resolve_input_shape(self) -> tuple[int | None, int, int, int]:
        shape = tuple(int(dim) for dim in get_tensor_shape(self.engine, self.input_name))
        if len(shape) != 4:
            raise EvaluationError(
                f"RF-DETR TensorRT backend expects NCHW input with rank 4, got shape {shape} for {self.input_name}."
            )

        batch, channels, height, width = shape
        if channels <= 0:
            channels = 3
        if channels not in {1, 3}:
            raise EvaluationError(f"Unsupported RF-DETR engine channel count {channels}; expected 1 or 3.")
        if height <= 0:
            height = self.imgsz
        if width <= 0:
            width = self.imgsz
        if height <= 0 or width <= 0:
            raise EvaluationError("TensorRT engine has dynamic H/W, so --imgsz must be a positive integer.")

        if shape[2] > 0 and shape[3] > 0 and self.imgsz and (height != self.imgsz or width != self.imgsz):
            raise EvaluationError(
                f"Engine input size is {height}x{width}, but --imgsz {self.imgsz} was requested. "
                f"Run with --imgsz {height} or rebuild the engine with the requested input size."
            )

        static_batch_size = batch if batch > 0 else None
        if static_batch_size is not None and self.requested_batch_size > static_batch_size:
            raise EvaluationError(
                f"Engine static batch is {static_batch_size}, but --batch-size {self.requested_batch_size} was requested."
            )
        return static_batch_size, channels, height, width

    def predict_batch(self, image_paths: Sequence[Path]) -> list[list[Prediction]]:
        if not image_paths:
            return []
        effective_paths = list(image_paths)
        if self.static_batch_size is not None:
            if len(effective_paths) > self.static_batch_size:
                raise EvaluationError(
                    f"Engine static batch is {self.static_batch_size}, received {len(effective_paths)} images."
                )
            while len(effective_paths) < self.static_batch_size:
                effective_paths.append(effective_paths[-1])

        batch_array, original_sizes = self._preprocess(effective_paths)
        input_tensor = torch.from_numpy(batch_array).to(self.device).contiguous()
        outputs = self._run_engine(input_tensor)
        decoded = self._decode_outputs(outputs, effective_paths, original_sizes)
        return decoded[: len(image_paths)]

    def _preprocess(self, image_paths: Sequence[Path]) -> tuple[np.ndarray, list[tuple[int, int]]]:
        batch = []
        original_sizes = []
        mean = np.array([0.485, 0.456, 0.406][: self.channels], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225][: self.channels], dtype=np.float32)
        pil_mode = "L" if self.channels == 1 else "RGB"

        for image_path in image_paths:
            with Image.open(image_path) as image:
                original_sizes.append(image.size)
                array = np.array(
                    image.convert(pil_mode).resize(
                        (self.input_width, self.input_height),
                        Image.Resampling.BILINEAR,
                    ),
                    dtype=np.float32,
                )
            array = array / 255.0
            if array.ndim == 2:
                array = array[:, :, np.newaxis]
            array = (array - mean) / std
            batch.append(array.transpose(2, 0, 1))

        np_dtype = np.float16 if self.input_dtype == np.dtype("float16") else np.float32
        return np.stack(batch, axis=0).astype(np_dtype), original_sizes

    def _run_engine(self, input_tensor: torch.Tensor) -> dict[str, np.ndarray]:
        input_shape = tuple(int(dim) for dim in input_tensor.shape)
        set_context_input_shape(self.context, self.engine, self.input_name, input_shape)

        tensors: dict[str, torch.Tensor] = {self.input_name: input_tensor}
        for output_name in self.output_names:
            shape = tuple(int(dim) for dim in get_context_tensor_shape(self.context, self.engine, output_name))
            if any(dim < 0 for dim in shape):
                raise EvaluationError(
                    f"Output tensor {output_name} still has dynamic shape {shape} after setting input shape {input_shape}."
                )
            dtype = torch_dtype_from_np_dtype(np.dtype(self.trt.nptype(get_tensor_dtype(self.engine, output_name))))
            tensors[output_name] = torch.empty(shape, dtype=dtype, device=self.device)

        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            if hasattr(self.context, "set_tensor_address") and hasattr(self.context, "execute_async_v3"):
                for name, tensor in tensors.items():
                    ok = self.context.set_tensor_address(name, int(tensor.data_ptr()))
                    if not ok:
                        raise EvaluationError(f"Failed to bind TensorRT tensor address for {name}.")
                ok = self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
            else:
                bindings = [int(tensors[name].data_ptr()) for name in self.tensor_names]
                ok = self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.cuda_stream)
            if not ok:
                raise EvaluationError("TensorRT execution failed.")
        self.stream.synchronize()
        return {name: tensors[name].detach().cpu().numpy() for name in self.output_names}

    def _decode_outputs(
        self,
        outputs: dict[str, np.ndarray],
        image_paths: Sequence[Path],
        original_sizes: Sequence[tuple[int, int]],
    ) -> list[list[Prediction]]:
        decoded = self._decode_postprocessed_outputs(outputs, image_paths, original_sizes)
        if decoded is not None:
            return decoded
        return self._decode_raw_rfdetr_outputs(outputs, image_paths, original_sizes)

    def _decode_postprocessed_outputs(
        self,
        outputs: dict[str, np.ndarray],
        image_paths: Sequence[Path],
        original_sizes: Sequence[tuple[int, int]],
    ) -> list[list[Prediction]] | None:
        boxes_name = find_output_name(outputs, ("box", "bbox"))
        scores_name = find_output_name(outputs, ("score", "conf"))
        classes_name = find_output_name(outputs, ("class", "label"))
        if boxes_name is None or scores_name is None or classes_name is None:
            return None

        boxes = ensure_batched(outputs[boxes_name])
        scores = ensure_batched(outputs[scores_name])
        classes = ensure_batched(outputs[classes_name])
        num_dets = outputs.get(find_output_name(outputs, ("num", "count")) or "", None)

        predictions: list[list[Prediction]] = []
        for batch_index, (image_path, (width, height)) in enumerate(zip(image_paths, original_sizes)):
            limit = int(num_dets.reshape(-1)[batch_index]) if num_dets is not None else boxes.shape[1]
            per_image = []
            for index in range(min(limit, boxes.shape[1])):
                score_value = scores[batch_index, index]
                if np.ndim(score_value) > 0:
                    score = float(np.max(score_value))
                else:
                    score = float(score_value)
                if score <= self.conf:
                    continue
                class_value = classes[batch_index, index]
                class_id = int(np.argmax(class_value)) if np.ndim(class_value) > 0 else int(class_value)
                box = tuple(float(v) for v in boxes[batch_index, index, :4])
                if max(abs(v) for v in box) <= 1.5:
                    box = (box[0] * width, box[1] * height, box[2] * width, box[3] * height)
                per_image.append(
                    Prediction(
                        image_id=image_id_for_path(image_path),
                        class_id=class_id,
                        confidence=score,
                        box=clamp_box(box, width, height),
                    )
                )
            predictions.append(per_image)
        return predictions

    def _decode_raw_rfdetr_outputs(
        self,
        outputs: dict[str, np.ndarray],
        image_paths: Sequence[Path],
        original_sizes: Sequence[tuple[int, int]],
    ) -> list[list[Prediction]]:
        boxes_name, logits_name = match_raw_rfdetr_outputs(outputs)
        if boxes_name is None or logits_name is None:
            shapes = {name: list(value.shape) for name, value in outputs.items()}
            raise EvaluationError(
                "Could not identify RF-DETR TensorRT outputs. Expected raw outputs named like "
                "`dets`/`labels` or one boxes tensor with last dim 4 plus one logits tensor. "
                f"Available outputs: {shapes}"
            )

        boxes_cxcywh = ensure_batched(outputs[boxes_name]).astype(np.float32)
        logits = ensure_batched(outputs[logits_name]).astype(np.float32)
        if logits.shape[-1] > 1:
            logits = logits[:, :, :-1]
        scores_all = sigmoid_np(logits)
        scores = scores_all.max(axis=-1)
        class_ids = scores_all.argmax(axis=-1)

        predictions: list[list[Prediction]] = []
        for batch_index, (image_path, (width, height)) in enumerate(zip(image_paths, original_sizes)):
            per_image = []
            keep = scores[batch_index] > self.conf
            for box_cxcywh, score, class_id in zip(
                boxes_cxcywh[batch_index][keep],
                scores[batch_index][keep],
                class_ids[batch_index][keep],
            ):
                cx, cy, box_width, box_height = (float(v) for v in box_cxcywh[:4])
                box = (
                    (cx - box_width / 2.0) * width,
                    (cy - box_height / 2.0) * height,
                    (cx + box_width / 2.0) * width,
                    (cy + box_height / 2.0) * height,
                )
                per_image.append(
                    Prediction(
                        image_id=image_id_for_path(image_path),
                        class_id=int(class_id),
                        confidence=float(score),
                        box=clamp_box(box, width, height),
                    )
                )
            predictions.append(per_image)
        return predictions


def import_tensorrt() -> Any:
    try:
        return importlib.import_module("tensorrt")
    except ImportError:
        try:
            return importlib.import_module("_tensorrt")
        except ImportError as exc:
            raise EvaluationError(
                "TensorRT Python bindings are not installed. Install NVIDIA TensorRT for the CUDA/TensorRT "
                "version used to build the .engine file."
            ) from exc


def engine_tensor_names(engine: Any) -> list[str]:
    if hasattr(engine, "num_io_tensors"):
        return [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]
    return [name for name in engine]


def tensor_is_input(engine: Any, trt: Any, name: str) -> bool:
    if hasattr(engine, "get_tensor_mode"):
        return engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
    if hasattr(engine, "binding_is_input"):
        try:
            return bool(engine.binding_is_input(name))
        except TypeError:
            return bool(engine.binding_is_input(engine.get_binding_index(name)))
    raise EvaluationError("Unsupported TensorRT engine API: cannot determine input/output tensor modes.")


def get_tensor_shape(engine: Any, name: str) -> tuple[int, ...]:
    if hasattr(engine, "get_tensor_shape"):
        return tuple(int(dim) for dim in engine.get_tensor_shape(name))
    if hasattr(engine, "get_binding_shape"):
        try:
            shape = engine.get_binding_shape(name)
        except TypeError:
            shape = engine.get_binding_shape(engine.get_binding_index(name))
        return tuple(int(dim) for dim in shape)
    raise EvaluationError("Unsupported TensorRT engine API: cannot read tensor shapes.")


def get_context_tensor_shape(context: Any, engine: Any, name: str) -> tuple[int, ...]:
    if hasattr(context, "get_tensor_shape"):
        return tuple(int(dim) for dim in context.get_tensor_shape(name))
    return get_tensor_shape(engine, name)


def get_tensor_dtype(engine: Any, name: str) -> Any:
    if hasattr(engine, "get_tensor_dtype"):
        return engine.get_tensor_dtype(name)
    if hasattr(engine, "get_binding_dtype"):
        try:
            return engine.get_binding_dtype(name)
        except TypeError:
            return engine.get_binding_dtype(engine.get_binding_index(name))
    raise EvaluationError("Unsupported TensorRT engine API: cannot read tensor dtypes.")


def set_context_input_shape(context: Any, engine: Any, name: str, shape: tuple[int, ...]) -> None:
    if hasattr(context, "set_input_shape"):
        ok = context.set_input_shape(name, shape)
    elif hasattr(context, "set_binding_shape"):
        ok = context.set_binding_shape(engine.get_binding_index(name), shape)
    else:
        ok = True
    if ok is False:
        raise EvaluationError(f"Failed to set TensorRT input shape for {name}: {shape}")


def torch_dtype_from_np_dtype(dtype: np.dtype) -> torch.dtype:
    if dtype == np.dtype("float16"):
        return torch.float16
    if dtype == np.dtype("float32"):
        return torch.float32
    if dtype == np.dtype("float64"):
        return torch.float64
    if dtype == np.dtype("int32"):
        return torch.int32
    if dtype == np.dtype("int64"):
        return torch.int64
    if dtype == np.dtype("int8"):
        return torch.int8
    if dtype == np.dtype("uint8"):
        return torch.uint8
    if dtype == np.dtype("bool"):
        return torch.bool
    raise EvaluationError(f"Unsupported TensorRT output dtype: {dtype}")


def find_output_name(outputs: dict[str, np.ndarray], patterns: Sequence[str]) -> str | None:
    lowered = {name: name.lower() for name in outputs}
    for pattern in patterns:
        match = next((name for name, lower in lowered.items() if pattern in lower), None)
        if match is not None:
            return match
    return None


def match_raw_rfdetr_outputs(outputs: dict[str, np.ndarray]) -> tuple[str | None, str | None]:
    boxes_name = find_output_name(outputs, ("dets", "pred_boxes", "boxes", "bbox"))
    logits_name = find_output_name(outputs, ("labels", "pred_logits", "logits"))
    if boxes_name is not None and logits_name is not None and boxes_name != logits_name:
        return boxes_name, logits_name

    names = list(outputs)
    rank3 = {name: ensure_batched(value) for name, value in outputs.items() if ensure_batched(value).ndim == 3}
    box_candidates = [name for name, value in rank3.items() if value.shape[-1] == 4]
    logit_candidates = [name for name, value in rank3.items() if name not in box_candidates or value.shape[-1] != 4]
    if len(box_candidates) == 1 and len(logit_candidates) == 1:
        return box_candidates[0], logit_candidates[0]

    if len(names) == 2:
        first, second = names
        first_value = ensure_batched(outputs[first])
        second_value = ensure_batched(outputs[second])
        if first_value.ndim == 3 and first_value.shape[-1] == 4:
            return first, second
        if second_value.ndim == 3 and second_value.shape[-1] == 4:
            return second, first
    return None, None


def ensure_batched(array: np.ndarray) -> np.ndarray:
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim == 2:
        return array[np.newaxis, ...]
    return array


def sigmoid_np(array: np.ndarray) -> np.ndarray:
    clipped = np.clip(array, -88, 88)
    return 1.0 / (1.0 + np.exp(-clipped))


def normalize_device_for_ultralytics(device: str) -> str | None:
    if device in {"", "auto", "none", "None"}:
        return None
    if device == "cuda":
        return "0"
    if device.startswith("cuda:"):
        return device.split(":", 1)[1]
    return device


def resolve_torch_device(device: str) -> torch.device:
    if device in {"", "auto"}:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        return torch.device("cuda:0")
    if device.isdigit():
        return torch.device(f"cuda:{device}")
    return torch.device(device)


def create_predictor(
    model_path: str | Path,
    backend: str,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    batch_size: int,
) -> BasePredictor:
    path = Path(model_path)
    selected = detect_backend(path) if backend == "auto" else backend
    if selected == "ultralytics":
        return UltralyticsPredictor(path, imgsz, conf, iou, device, batch_size)
    if selected == "rfdetr":
        return RFDETRPredictor(path, imgsz, conf, iou, device, batch_size)
    if selected == "rfdetr_engine":
        return RFDETRTensorRTPredictor(path, imgsz, conf, iou, device, batch_size)
    raise EvaluationError(f"Unsupported backend `{selected}` for model {path}")


def detect_backend(model_path: Path) -> str:
    suffix = model_path.suffix.lower()
    if suffix == ".pth":
        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict) and (
                "rfdetr_version" in checkpoint or str(checkpoint.get("model_name", "")).lower().startswith("rfdetr")
            ):
                return "rfdetr"
        except Exception:
            pass
    if suffix == ".engine":
        return "rfdetr_engine"
    return "ultralytics"


def load_dataset_samples(data: str | Path, split: str = "val", max_images: int | None = None) -> list[DatasetSample]:
    data_path = Path(data)
    if not data_path.exists():
        raise EvaluationError(f"Dataset path does not exist: {data_path}")

    if data_path.suffix.lower() in {".yaml", ".yml"}:
        image_paths, annotation_path = _paths_from_yaml(data_path, split)
    elif data_path.suffix.lower() == ".json":
        image_paths, annotation_path = _paths_from_coco_json(data_path)
    elif data_path.is_file():
        image_paths = _image_paths_from_text_file(data_path)
        annotation_path = None
    else:
        image_paths, annotation_path = _paths_from_dataset_dir(data_path, split)

    image_paths = sorted(dict.fromkeys(path.resolve() for path in image_paths))
    if max_images and max_images > 0:
        image_paths = image_paths[:max_images]
    if not image_paths:
        raise EvaluationError(f"No images found for split `{split}` in {data_path}")

    targets_by_image = _load_targets(image_paths, annotation_path)
    samples: list[DatasetSample] = []
    for path in image_paths:
        width, height = read_image_size(path)
        image_id = image_id_for_path(path)
        samples.append(
            DatasetSample(
                image_id=image_id,
                image_path=path,
                width=width,
                height=height,
                targets=tuple(targets_by_image.get(image_id, ())),
            )
        )
    return samples


def _paths_from_yaml(yaml_path: Path, split: str) -> tuple[list[Path], Path | None]:
    with yaml_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if split not in config:
        fallback = "val" if "val" in config else "valid" if "valid" in config else "test" if "test" in config else None
        if fallback is None:
            raise EvaluationError(f"`{yaml_path}` has no `{split}`, `val`, or `test` entry.")
        split = fallback

    base = Path(config.get("path", yaml_path.parent))
    if not base.is_absolute():
        base = (yaml_path.parent / base).resolve()

    split_value = config[split]
    image_paths = _resolve_image_source(split_value, base)
    annotation_path = _find_coco_annotation_for_images(image_paths, base, split)
    return image_paths, annotation_path


def _paths_from_coco_json(json_path: Path) -> tuple[list[Path], Path]:
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    base = json_path.parent
    image_paths = []
    for image in payload.get("images", []):
        file_name = image.get("file_name")
        if not file_name:
            continue
        candidate = (base / file_name).resolve()
        if not candidate.exists():
            candidate = _find_file_by_name(base, Path(file_name).name)
        if candidate and candidate.exists():
            image_paths.append(candidate)
    return image_paths, json_path


def _paths_from_dataset_dir(dataset_dir: Path, split: str) -> tuple[list[Path], Path | None]:
    split_names = split_aliases(split)
    candidates = [
        *(dataset_dir / name / "images" for name in split_names),
        *(dataset_dir / name for name in split_names),
        *(dataset_dir / "images" / name for name in split_names),
        dataset_dir / "images",
        dataset_dir,
    ]
    image_dir = next((candidate for candidate in candidates if candidate.exists() and candidate.is_dir()), None)
    if image_dir is None:
        raise EvaluationError(f"No image directory found under {dataset_dir}")

    image_paths = list(_iter_images(image_dir))
    annotation_path = _find_coco_annotation_for_images(image_paths, dataset_dir, split)
    return image_paths, annotation_path


def split_aliases(split: str) -> list[str]:
    if split == "val":
        return ["val", "valid", "validation"]
    if split == "valid":
        return ["valid", "val", "validation"]
    if split == "test":
        return ["test", "testing"]
    return [split]


def _resolve_image_source(value: Any, base: Path) -> list[Path]:
    if isinstance(value, (list, tuple)):
        paths: list[Path] = []
        for item in value:
            paths.extend(_resolve_image_source(item, base))
        return paths

    source = Path(str(value))
    if not source.is_absolute():
        source = (base / source).resolve()

    if source.is_file() and source.suffix.lower() in {".txt", ".csv"}:
        return _image_paths_from_text_file(source, base=base)
    if source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
        return [source]
    if source.is_dir():
        return list(_iter_images(source))
    raise EvaluationError(f"Unable to resolve image source: {source}")


def _image_paths_from_text_file(path: Path, base: Path | None = None) -> list[Path]:
    root = base or path.parent
    image_paths = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip().split(",", 1)[0]
            if not stripped:
                continue
            image_path = Path(stripped)
            if not image_path.is_absolute():
                image_path = (root / image_path).resolve()
            if image_path.exists():
                image_paths.append(image_path)
    return image_paths


def _iter_images(directory: Path) -> Iterable[Path]:
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _find_coco_annotation_for_images(image_paths: Sequence[Path], base: Path, split: str) -> Path | None:
    split_names = split_aliases(split)
    candidates = [
        *(base / name / "_annotations.coco.json" for name in split_names),
        *(base / name / "_annotations.json" for name in split_names),
        *(base / "annotations" / f"instances_{name}.json" for name in split_names),
        *(base / "annotations" / f"instances_{name}2017.json" for name in split_names),
        *(base / f"{name}.json" for name in split_names),
        base / "_annotations.coco.json",
    ]
    if image_paths:
        first_parent = image_paths[0].parent
        candidates.extend(
            [
                first_parent.parent / "_annotations.coco.json",
                first_parent.parent / "_annotations.json",
                first_parent / "_annotations.coco.json",
            ]
        )
    return next((candidate for candidate in candidates if candidate.exists()), None)


def _load_targets(image_paths: Sequence[Path], annotation_path: Path | None) -> dict[str, list[GroundTruth]]:
    if annotation_path is not None:
        return _load_coco_targets(annotation_path, image_paths)
    return _load_yolo_targets(image_paths)


def _load_coco_targets(annotation_path: Path, image_paths: Sequence[Path]) -> dict[str, list[GroundTruth]]:
    with annotation_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    by_name = {path.name: path for path in image_paths}
    by_stem = {path.stem: path for path in image_paths}
    image_id_to_path: dict[int, Path] = {}
    for image in payload.get("images", []):
        file_name = Path(str(image.get("file_name", ""))).name
        candidate = by_name.get(file_name) or by_stem.get(Path(file_name).stem)
        if candidate is not None:
            image_id_to_path[int(image["id"])] = candidate

    categories = sorted(payload.get("categories", []), key=lambda item: int(item["id"]))
    category_to_contiguous = {int(category["id"]): index for index, category in enumerate(categories)}

    targets: dict[str, list[GroundTruth]] = {}
    for annotation in payload.get("annotations", []):
        if annotation.get("iscrowd", 0):
            continue
        path = image_id_to_path.get(int(annotation.get("image_id", -1)))
        if path is None:
            continue
        bbox = annotation.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x, y, width, height = (float(v) for v in bbox[:4])
        box = (x, y, x + width, y + height)
        class_id = category_to_contiguous.get(int(annotation["category_id"]), int(annotation["category_id"]))
        image_id = image_id_for_path(path)
        targets.setdefault(image_id, []).append(GroundTruth(image_id=image_id, class_id=class_id, box=box))
    return targets


def _load_yolo_targets(image_paths: Sequence[Path]) -> dict[str, list[GroundTruth]]:
    targets: dict[str, list[GroundTruth]] = {}
    for image_path in image_paths:
        label_path = infer_yolo_label_path(image_path)
        if label_path is None or not label_path.exists():
            continue
        width, height = read_image_size(image_path)
        image_id = image_id_for_path(image_path)
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(float(parts[0]))
                coords = [float(value) for value in parts[1:]]
            except ValueError:
                continue
            box = yolo_coords_to_xyxy(coords, width, height)
            if box is None:
                continue
            targets.setdefault(image_id, []).append(GroundTruth(image_id=image_id, class_id=class_id, box=box))
    return targets


def infer_yolo_label_path(image_path: Path) -> Path | None:
    parts = list(image_path.parts)
    if "images" in parts:
        index = len(parts) - 1 - parts[::-1].index("images")
        label_parts = parts[:]
        label_parts[index] = "labels"
        return Path(*label_parts).with_suffix(".txt")

    parent = image_path.parent
    candidates = [
        parent / "labels" / f"{image_path.stem}.txt",
        parent.parent / "labels" / f"{image_path.stem}.txt",
    ]
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def yolo_coords_to_xyxy(coords: Sequence[float], width: int, height: int) -> tuple[float, float, float, float] | None:
    if len(coords) == 4:
        cx, cy, box_width, box_height = coords
        x1 = (cx - box_width / 2.0) * width
        y1 = (cy - box_height / 2.0) * height
        x2 = (cx + box_width / 2.0) * width
        y2 = (cy + box_height / 2.0) * height
    elif len(coords) >= 6 and len(coords) % 2 == 0:
        xs = coords[0::2]
        ys = coords[1::2]
        x1 = min(xs) * width
        y1 = min(ys) * height
        x2 = max(xs) * width
        y2 = max(ys) * height
    else:
        return None
    return clamp_box((x1, y1, x2, y2), width, height)


def clamp_box(
    box: tuple[float, float, float, float],
    width: int | float,
    height: int | float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    )


def read_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def image_id_for_path(path: Path) -> str:
    return str(path.resolve())


def _find_file_by_name(root: Path, name: str) -> Path | None:
    matches = list(root.rglob(name))
    return matches[0] if matches else None


def evaluate_predictions(
    samples: Sequence[DatasetSample],
    predictions: Sequence[Prediction],
    iou_thresholds: Sequence[float] = IOU_THRESHOLDS,
    metric_backend: str = "coco",
    operating_conf: float | None = None,
) -> dict[str, float]:
    if metric_backend == "coco":
        return evaluate_predictions_coco(samples, predictions, operating_conf=operating_conf)
    if metric_backend != "local":
        raise EvaluationError(f"Unsupported metric backend: {metric_backend}")
    return evaluate_predictions_local(samples, predictions, iou_thresholds)


def evaluate_predictions_coco(
    samples: Sequence[DatasetSample],
    predictions: Sequence[Prediction],
    operating_conf: float | None = None,
) -> dict[str, float]:
    """Evaluate AP with the official COCO API and operating-point P/R locally."""
    ground_truths = [target for sample in samples for target in sample.targets]
    empty_metrics = {
        "precision": math.nan,
        "recall": math.nan,
        "map50": math.nan,
        "map50_95": math.nan,
        "num_images": float(len(samples)),
        "num_labels": float(len(ground_truths)),
        "num_predictions": float(len(predictions)),
    }
    if not ground_truths:
        return empty_metrics

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise EvaluationError(
            "pycocotools is required for standard COCO metrics. "
            "Install it or run with --metric-backend local."
        ) from exc

    image_ids = {sample.image_id: index + 1 for index, sample in enumerate(samples)}
    class_ids = sorted(
        {target.class_id for target in ground_truths}
        | {prediction.class_id for prediction in predictions}
    )
    category_ids = {class_id: index + 1 for index, class_id in enumerate(class_ids)}
    coco_dataset = {
        "info": {"description": "Generated evaluation dataset"},
        "licenses": [],
        "images": [
            {
                "id": image_ids[sample.image_id],
                "file_name": str(sample.image_path),
                "width": sample.width,
                "height": sample.height,
            }
            for sample in samples
        ],
        "categories": [
            {"id": category_id, "name": str(class_id)}
            for class_id, category_id in category_ids.items()
        ],
        "annotations": [
            {
                "id": index + 1,
                "image_id": image_ids[target.image_id],
                "category_id": category_ids[target.class_id],
                "bbox": xyxy_to_xywh(target.box),
                "area": box_area(target.box),
                "iscrowd": 0,
            }
            for index, target in enumerate(ground_truths)
        ],
    }
    coco_results = [
        {
            "image_id": image_ids[prediction.image_id],
            "category_id": category_ids[prediction.class_id],
            "bbox": xyxy_to_xywh(prediction.box),
            "score": prediction.confidence,
        }
        for prediction in predictions
        if prediction.image_id in image_ids and prediction.class_id in category_ids
    ]

    coco_ground_truth = COCO()
    coco_ground_truth.dataset = coco_dataset
    with contextlib.redirect_stdout(io.StringIO()):
        coco_ground_truth.createIndex()
    if not coco_results:
        return {**empty_metrics, "map50": 0.0, "map50_95": 0.0}

    with contextlib.redirect_stdout(io.StringIO()):
        coco_detections = coco_ground_truth.loadRes(coco_results)
        evaluator = COCOeval(coco_ground_truth, coco_detections, iouType="bbox")
        evaluator.params.imgIds = list(image_ids.values())
        evaluator.params.catIds = list(category_ids.values())
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    operating_predictions = (
        [prediction for prediction in predictions if prediction.confidence >= operating_conf]
        if operating_conf is not None
        else list(predictions)
    )
    _, precision_50, recall_50 = _compute_ap_at_iou(
        ground_truths,
        operating_predictions,
        0.50,
    )
    return {
        "precision": precision_50,
        "recall": recall_50,
        "map50": float(evaluator.stats[1]),
        "map50_95": float(evaluator.stats[0]),
        "num_images": float(len(samples)),
        "num_labels": float(len(ground_truths)),
        "num_predictions": float(len(predictions)),
    }


def xyxy_to_xywh(box: Sequence[float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), max(0.0, float(x2 - x1)), max(0.0, float(y2 - y1))]


def box_area(box: Sequence[float]) -> float:
    _, _, width, height = xyxy_to_xywh(box)
    return width * height


def evaluate_predictions_local(
    samples: Sequence[DatasetSample],
    predictions: Sequence[Prediction],
    iou_thresholds: Sequence[float] = IOU_THRESHOLDS,
) -> dict[str, float]:
    ground_truths = [target for sample in samples for target in sample.targets]
    if not ground_truths:
        return {
            "precision": math.nan,
            "recall": math.nan,
            "map50": math.nan,
            "map50_95": math.nan,
            "num_images": float(len(samples)),
            "num_labels": 0.0,
            "num_predictions": float(len(predictions)),
        }

    ap_by_threshold = []
    precision_50 = math.nan
    recall_50 = math.nan
    for threshold in iou_thresholds:
        ap, precision, recall = _compute_ap_at_iou(ground_truths, predictions, threshold)
        ap_by_threshold.append(ap)
        if abs(threshold - 0.50) < 1e-9:
            precision_50 = precision
            recall_50 = recall

    map50 = ap_by_threshold[0] if ap_by_threshold else math.nan
    map5095 = float(np.nanmean(ap_by_threshold)) if ap_by_threshold else math.nan
    return {
        "precision": precision_50,
        "recall": recall_50,
        "map50": map50,
        "map50_95": map5095,
        "num_images": float(len(samples)),
        "num_labels": float(len(ground_truths)),
        "num_predictions": float(len(predictions)),
    }


def _compute_ap_at_iou(
    ground_truths: Sequence[GroundTruth],
    predictions: Sequence[Prediction],
    iou_threshold: float,
) -> tuple[float, float, float]:
    classes = sorted({target.class_id for target in ground_truths} | {prediction.class_id for prediction in predictions})
    aps: list[float] = []
    total_tp = 0
    total_fp = 0
    total_gt = 0

    for class_id in classes:
        class_gts = [target for target in ground_truths if target.class_id == class_id]
        class_predictions = sorted(
            [prediction for prediction in predictions if prediction.class_id == class_id],
            key=lambda prediction: prediction.confidence,
            reverse=True,
        )
        total_gt += len(class_gts)
        if not class_gts:
            total_fp += len(class_predictions)
            continue

        gt_by_image: dict[str, list[GroundTruth]] = {}
        for target in class_gts:
            gt_by_image.setdefault(target.image_id, []).append(target)
        matched = {image_id: np.zeros(len(items), dtype=bool) for image_id, items in gt_by_image.items()}

        true_positives = np.zeros(len(class_predictions), dtype=float)
        false_positives = np.zeros(len(class_predictions), dtype=float)

        for index, prediction in enumerate(class_predictions):
            candidates = gt_by_image.get(prediction.image_id, [])
            if not candidates:
                false_positives[index] = 1.0
                continue

            ious = np.array([box_iou(prediction.box, target.box) for target in candidates], dtype=float)
            best_index = int(np.argmax(ious)) if len(ious) else -1
            best_iou = float(ious[best_index]) if best_index >= 0 else 0.0

            if best_iou >= iou_threshold and not matched[prediction.image_id][best_index]:
                true_positives[index] = 1.0
                matched[prediction.image_id][best_index] = True
            else:
                false_positives[index] = 1.0

        total_tp += int(true_positives.sum())
        total_fp += int(false_positives.sum())

        if len(class_predictions) == 0:
            aps.append(0.0)
            continue

        tp_cumsum = np.cumsum(true_positives)
        fp_cumsum = np.cumsum(false_positives)
        recalls = tp_cumsum / max(len(class_gts), np.finfo(float).eps)
        precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(float).eps)
        aps.append(_compute_101_point_ap(recalls, precisions))

    precision = total_tp / max(total_tp + total_fp, np.finfo(float).eps)
    recall = total_tp / max(total_gt, np.finfo(float).eps)
    return float(np.mean(aps)) if aps else math.nan, float(precision), float(recall)


def _compute_101_point_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    recall_grid = np.linspace(0.0, 1.0, 101)
    if len(recalls) == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for index in range(len(mpre) - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])

    values = []
    for recall in recall_grid:
        indices = np.where(mrec >= recall)[0]
        values.append(mpre[indices[0]] if indices.size else 0.0)
    return float(np.mean(values))


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def evaluate_model(
    model_path: str | Path,
    data: str | Path,
    imgsz: int,
    batch_size: int,
    conf: float,
    iou: float,
    device: str,
    split: str = "val",
    backend: str = "auto",
    max_images: int | None = None,
    metric_backend: str = "coco",
) -> dict[str, float]:
    samples = load_dataset_samples(data, split=split, max_images=max_images)
    predictor = create_predictor(model_path, backend, imgsz, conf, iou, device, batch_size)
    try:
        return evaluate_with_predictor(
            predictor,
            samples,
            batch_size=batch_size,
            metric_backend=metric_backend,
        )
    finally:
        predictor.close()


def evaluate_with_predictor(
    predictor: BasePredictor,
    samples: Sequence[DatasetSample],
    batch_size: int,
    metric_backend: str = "coco",
) -> dict[str, float]:
    predictions: list[Prediction] = []
    for batch in batched(samples, batch_size):
        image_paths = [sample.image_path for sample in batch]
        for per_image in predictor.predict_batch(image_paths):
            predictions.extend(per_image)
        synchronize_if_cuda()
    return evaluate_predictions(samples, predictions, metric_backend=metric_backend)


def batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def synchronize_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def write_evaluation_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate precision, recall, mAP50, and mAP50-95.")
    parser.add_argument("--model", required=True, help="Model checkpoint path.")
    parser.add_argument("--data", required=True, help="Dataset YAML, dataset directory, COCO JSON, or image list.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold passed to the model backend.")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, cuda:0, or an Ultralytics device id.")
    parser.add_argument("--split", default="val", help="Dataset split to evaluate: val, valid, or test.")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "ultralytics", "rfdetr", "rfdetr_engine"],
        help="Model backend.",
    )
    parser.add_argument("--max-images", type=int, default=0, help="Optional limit for quick checks. 0 means all images.")
    parser.add_argument(
        "--metric-backend",
        choices=["coco", "local"],
        default="coco",
        help="AP implementation. 'coco' uses the official pycocotools evaluator.",
    )
    parser.add_argument("--output", default="", help="Optional CSV path to append one evaluation row.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_model(
        model_path=args.model,
        data=args.data,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        split=args.split,
        backend=args.backend,
        max_images=args.max_images or None,
        metric_backend=args.metric_backend,
    )
    row = {
        "Model": Path(args.model).stem,
        "Model Path": args.model,
        "Dataset": args.data,
        "Split": args.split,
        "Image Size": args.imgsz,
        "Batch Size": args.batch_size,
        "Confidence": args.conf,
        "IoU": args.iou,
        "Metric Backend": args.metric_backend,
        "Device": args.device,
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "mAP50": metrics["map50"],
        "mAP50-95": metrics["map50_95"],
        "Images": int(metrics["num_images"]),
        "Labels": int(metrics["num_labels"]),
        "Predictions": int(metrics["num_predictions"]),
    }
    if args.output:
        write_evaluation_csv(Path(args.output), row)
    print(json.dumps(row, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
