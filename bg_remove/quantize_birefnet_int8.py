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
  - Preprocessing matches app inference: RGB -> resize 512x512 -> float32 -> normalize by mean/std -> NCHW.
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


IMAGE_SIZE = (512, 512)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(path: Optional[Path]) -> np.ndarray:
    """Load image (or generate synthetic), apply BiRefNet preprocessing, return NCHW float32."""
    if path is None:
        # synthetic RGB image with simple structure
        arr = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.uint8)
        # draw simple gradients/patches
        for c in range(3):
            arr[:, :, c] = np.linspace(0, 255, IMAGE_SIZE[0], dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
    else:
        img = Image.open(path).convert("RGB")
        img = img.resize(IMAGE_SIZE, Image.LANCZOS)

    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    x = arr.transpose(2, 0, 1)[None]  # (1,3,H,W)
    return x


def discover_model_input_name(model_path: Path) -> str:
    model = onnx.load(str(model_path))
    graph = model.graph
    if not graph.input:
        raise RuntimeError("ONNX model has no graph input")
    # Prefer first input tensor name
    return graph.input[0].name


class ImageFolderDataReader(CalibrationDataReader):
    def __init__(
        self,
        input_name: str,
        calib_dir: Optional[Path],
        num_samples: int,
        exts: Optional[List[str]] = None,
    ) -> None:
        self.input_name = input_name
        self.calib_dir = calib_dir
        self.num_samples = num_samples
        self.exts = exts or [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
        self._iter: Optional[Iterator[np.ndarray]] = None

        self.files: List[Optional[Path]] = []
        if calib_dir and calib_dir.exists():
            all_files = [p for p in calib_dir.rglob("*") if p.suffix.lower() in self.exts]
            random.shuffle(all_files)
            self.files = all_files[:num_samples]
        else:
            # fallback to synthetic
            self.files = [None] * num_samples

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self._iter is None:
            self._iter = (preprocess_image(p) for p in self.files)
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
        default=None,
        help="Limit op types to quantize (e.g., Conv MatMul) to reduce calibration memory",
    )
    args = parser.parse_args()

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

        input_name = discover_model_input_name(args.input_model)
        reader = ImageFolderDataReader(input_name, args.calib_dir, args.num_calib)

        method = CalibrationMethod.Entropy if args.method == "entropy" else CalibrationMethod.MinMax

        # Use QDQ format which tends to preserve accuracy better on ORT
        # Pass optional session/provider args only if supported by installed onnxruntime version
        qs_sig = inspect.signature(quantize_static)
        qs_kwargs = dict(
            model_input=str(args.input_model),
            model_output=str(args.output_model),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
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

        quantize_static(**qs_kwargs)
        print(f"Saved statically-quantized (INT8) model to: {args.output_model}")


if __name__ == "__main__":
    main()


