#!/usr/bin/env python3
"""
INT8 post-training quantization for BiRefNet-lite ONNX (512x512).

Usage:
  # Static PTQ (calibrated, higher accuracy; uses more RAM)
  python tools/quantize_birefnet_int8.py \
    --input-model /path/to/BiRefNet_lite-matting.onnx \
    --output-model /path/to/BiRefNet_lite-matting-int8.onnx \
    --calib-dir /path/to/calibration_images \
    --num-calib 50 --method entropy --per-channel

  # Dynamic (weight-only) quantization (low RAM, no calibration)
  python tools/quantize_birefnet_int8.py \
    --input-model /path/to/BiRefNet_lite-matting.onnx \
    --output-model /path/to/BiRefNet_lite-matting-int8.onnx \
    --dynamic --per-channel

Notes:
  - If --calib-dir is omitted, synthetic images are used (works but real faces yield better accuracy).
  - Preprocessing matches app inference: RGB -> resize to model HxW -> float32 -> normalize by mean/std -> NCHW.
  - Static PTQ feeds one image per ORT run (batch size 1). ORT still buffers activations for every
    image in a stride chunk before building histograms; use --calib-stride 5 (or 10) to cap peak RAM
    when --num-calib is large (e.g. 50).
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np
import onnx
from PIL import Image
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
    quantize_dynamic,
)
import inspect
import onnxruntime as ort


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def discover_model_input_shape(model_path: Path) -> tuple[str, tuple[int, int]]:
    model = onnx.load(str(model_path))
    graph = model.graph
    if not graph.input:
        raise RuntimeError("ONNX model has no graph input")
    inp = graph.input[0]
    dims = inp.type.tensor_type.shape.dim
    if len(dims) < 4:
        raise RuntimeError(f"Expected NCHW input, got {len(dims)} dims")
    h = dims[2].dim_value
    w = dims[3].dim_value
    if h <= 0 or w <= 0:
        raise RuntimeError(
            "Input height/width are not fixed in the ONNX graph; "
            "re-export with --no-dynamic and explicit --height/--width."
        )
    return inp.name, (h, w)


def preprocess_image(path: Optional[Path], image_size: tuple[int, int]) -> np.ndarray:
    """Load image (or generate synthetic), apply BiRefNet preprocessing, return NCHW float32."""
    if path is None:
        # synthetic RGB image with simple structure
        arr = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        # draw simple gradients/patches
        for c in range(3):
            arr[:, :, c] = np.linspace(0, 255, image_size[0], dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
    else:
        img = Image.open(path).convert("RGB")
        img = img.resize(image_size, Image.LANCZOS)

    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    x = arr.transpose(2, 0, 1)[None]  # (1,3,H,W)
    return x


class ImageFolderDataReader(CalibrationDataReader):
    def __init__(
        self,
        input_name: str,
        image_size: tuple[int, int],
        calib_dir: Optional[Path],
        num_samples: int,
        exts: Optional[List[str]] = None,
    ) -> None:
        self.input_name = input_name
        self.image_size = image_size
        self.calib_dir = calib_dir
        self.num_samples = num_samples
        self.exts = exts or [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
        self._iter: Optional[Iterator[np.ndarray]] = None
        self._start = 0
        self._end = num_samples

        self.files: List[Optional[Path]] = []
        if calib_dir and calib_dir.exists():
            all_files = [p for p in calib_dir.rglob("*") if p.suffix.lower() in self.exts]
            random.shuffle(all_files)
            self.files = all_files[:num_samples]
        else:
            # fallback to synthetic
            self.files = [None] * num_samples

    def __len__(self) -> int:
        return len(self.files)

    def set_range(self, start_index: int, end_index: int) -> None:
        self._start = start_index
        self._end = end_index
        self.rewind()

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self._iter is None:
            self._iter = (
                preprocess_image(p, self.image_size)
                for p in self.files[self._start : self._end]
            )
        try:
            batch = next(self._iter)
            return {self.input_name: batch}
        except StopIteration:
            return None

    def rewind(self) -> None:
        self._iter = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize BiRefNet-lite ONNX to INT8 (PTQ)")
    parser.add_argument("--input-model", type=Path, required=True, help="Path to FP32 ONNX model")
    parser.add_argument("--output-model", type=Path, required=True, help="Path to write INT8 ONNX model")
    parser.add_argument("--calib-dir", type=Path, default=None, help="Directory of calibration images")
    parser.add_argument("--num-calib", type=int, default=50, help="Number of calibration samples (static PTQ)")
    parser.add_argument(
        "--calib-stride",
        type=int,
        default=5,
        help="Process this many calibration images per ORT pass (lowers peak RAM). "
        "Set to 0 to load all --num-calib images at once (may OOM on large models).",
    )
    parser.add_argument("--method", type=str, default="entropy", choices=["entropy", "minmax"], help="Calibration method")
    parser.add_argument("--per-channel", action="store_true", help="Enable per-channel quantization where supported")
    parser.add_argument("--dynamic", action="store_true", help="Use dynamic (weight-only) quantization to minimize RAM and skip calibration")
    # Performance/memory controls for ONNX Runtime during calibration
    parser.add_argument("--threads", type=int, default=1, help="ONNX Runtime intra-op threads (1 reduces memory fragmentation)")
    parser.add_argument("--sequential", action="store_true", help="Force sequential execution (may lower peak memory)")
    parser.add_argument("--disable-mem-pattern", action="store_true", help="Disable memory pattern (can reduce large contiguous allocations)")
    parser.add_argument("--disable-mem-arena", action="store_true", help="Disable CPU memory arena (fallback allocator can help OOM)")
    parser.add_argument(
        "--graph-opt",
        type=str,
        choices=["disable", "basic", "extended", "all"],
        default="extended",
        help="Graph optimization level during calibration",
    )
    parser.add_argument(
        "--ops",
        nargs="+",
        default=["Conv", "MatMul"],
        help="Limit op types to quantize (e.g., Conv MatMul) to reduce calibration memory",
    )
    args = parser.parse_args()

    input_name, image_size = discover_model_input_shape(args.input_model)
    print(f"[info] Model input: {input_name}, size={image_size[0]}x{image_size[1]}")

    if args.dynamic:
        # Low-memory path: weight-only quantization; activations remain fp32
        quantize_dynamic(
            model_input=str(args.input_model),
            model_output=str(args.output_model),
            weight_type=QuantType.QInt8,
            per_channel=args.per_channel,
            op_types_to_quantize=["Conv", "MatMul"],
        )
        print(f"Saved dynamically-quantized (INT8 weights) model to: {args.output_model}")
    else:
        # Configure ONNX Runtime session options to lower memory pressure during calibration inference
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, int(args.threads))
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)   
        os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"
        if args.sequential:
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # Graph optimization level
        if args.graph_opt == "disable":
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        elif args.graph_opt == "basic":
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        elif args.graph_opt == "extended":
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        else:
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Memory knobs
        if args.disable_mem_pattern:
            so.enable_mem_pattern = False
        if args.disable_mem_arena:
            so.enable_cpu_mem_arena = False

        reader = ImageFolderDataReader(
            input_name, image_size, args.calib_dir, args.num_calib
        )
        if args.calib_dir and reader.files and reader.files[0] is not None:
            print(f"[info] Calibration images: {len(reader.files)} from {args.calib_dir}")
        else:
            print(f"[info] Calibration images: {len(reader.files)} synthetic")

        method = CalibrationMethod.Entropy if args.method == "entropy" else CalibrationMethod.MinMax

        # Use QDQ format which tends to preserve accuracy better on ORT
        # Pass optional session/provider args only if supported by installed onnxruntime version
        qs_sig = inspect.signature(quantize_static)
        qs_kwargs = dict(
            model_input=str(args.input_model),
            model_output=str(args.output_model),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=method,
            per_channel=args.per_channel,
        )
        if args.ops is not None and "op_types_to_quantize" in qs_sig.parameters:
            qs_kwargs["op_types_to_quantize"] = args.ops
        if "session_options" in qs_sig.parameters:
            qs_kwargs["session_options"] = so
        if "providers" in qs_sig.parameters:
            qs_kwargs["providers"] = ["CPUExecutionProvider"]
        if "provider_options" in qs_sig.parameters:
            qs_kwargs["provider_options"] = None
        # Prefer asymmetric activations (UInt8) and symmetric weights (Int8) if supported
        extra_options: dict = {
            "ActivationSymmetric": False,
            "WeightSymmetric": True,
            "EnableSubgraph": True,
        }
        stride = int(args.calib_stride)
        if stride > 0:
            if len(reader) % stride != 0:
                raise SystemExit(
                    f"[error] --num-calib ({len(reader)}) must be divisible by --calib-stride ({stride})."
                )
            extra_options["CalibStridedMinMax"] = stride
            print(
                f"[info] Strided calibration: {len(reader)} images in chunks of {stride} "
                f"({len(reader) // stride} passes)"
            )
        elif len(reader) > 10:
            print(
                "[warn] --calib-stride 0 loads all calibration activations before histogramming; "
                "expect high RAM use on 512x512 models."
            )
        if "extra_options" in qs_sig.parameters:
            qs_kwargs["extra_options"] = extra_options

        quantize_static(**qs_kwargs)
        print(f"Saved statically-quantized (INT8) model to: {args.output_model}")


if __name__ == "__main__":
    main()


