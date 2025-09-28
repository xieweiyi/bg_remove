#!/usr/bin/env python3
"""
INT8 post-training quantization for GFPGAN ONNX using ONNX Runtime.

Supports both calibrated static PTQ (QDQ) and dynamic (weight-only) quantization.

Preprocessing matches the runtime path you shared:
- Start with RGB image
- Resize to configurable square size (default 512)
- Scale to [0,1], normalize to [-1,1] per channel
- Convert to NCHW float32 with batch dimension

Usage examples:
  # Static PTQ with calibration images (recommended for accuracy)
  python bg_remove/quantize_gfpgan_int8.py \
    --input-model d:/models/GFPGANv1.4.onnx \
    --output-model d:/models/GFPGANv1.4-int8.onnx \
    --calib-dir d:/data/faces \
    --num-calib 100 --method entropy --per-channel

  # Dynamic (weight-only) quantization (no calibration)
  python bg_remove/quantize_gfpgan_int8.py \
    --input-model d:/models/GFPGANv1.4.onnx \
    --output-model d:/models/GFPGANv1.4-int8-dynamic.onnx \
    --dynamic --per-channel
"""

from __future__ import annotations

import argparse
import random
import inspect
from pathlib import Path
import os
import tempfile
from typing import Dict, Iterator, List, Optional

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)


IMAGE_SIZE = (512, 512)
# Calibration-time resize (can be lowered to reduce memory usage)
CALIB_IMAGE_SIZE = (512, 512)


def _load_image_rgb(path: Optional[Path]) -> np.ndarray:
    if path is None:
        # Synthetic RGB image: simple gradients (calibration fallback)
        h, w = CALIB_IMAGE_SIZE[1], CALIB_IMAGE_SIZE[0]
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        for c in range(3):
            arr[:, :, c] = np.linspace(0, 255, w, dtype=np.uint8)
        return arr
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def preprocess_image_gfpgan(path: Optional[Path]) -> np.ndarray:
    """Preprocess image to NCHW float32 as used by your GFPGAN ONNX path."""
    rgb = _load_image_rgb(path)
    img = np.array(rgb, dtype=np.float32)
    img = cv2.resize(img, CALIB_IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    img = img / 255.0
    img = (img - 0.5) / 0.5
    img = np.ascontiguousarray(img.transpose(2, 0, 1))[np.newaxis, ...]
    return img


def discover_model_input_name(model_path: Path) -> str:
    model = onnx.load(str(model_path))
    graph = model.graph
    if not graph.input:
        raise RuntimeError("ONNX model has no graph input")
    return graph.input[0].name


class FaceFolderDataReader(CalibrationDataReader):
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
            self.files = [None] * num_samples

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self._iter is None:
            self._iter = (preprocess_image_gfpgan(p) for p in self.files)
        try:
            batch = next(self._iter)
            return {self.input_name: batch}
        except StopIteration:
            return None

    def rewind(self) -> None:
        self._iter = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize GFPGAN ONNX to INT8 (PTQ)")
    parser.add_argument("--input-model", type=Path, required=True, help="Path to FP32 ONNX model")
    parser.add_argument("--output-model", type=Path, required=True, help="Path to write INT8 ONNX model")
    parser.add_argument("--calib-dir", type=Path, default=None, help="Directory of calibration images")
    parser.add_argument("--num-calib", type=int, default=50, help="Number of calibration samples (static PTQ)")
    parser.add_argument(
        "--calib-size",
        type=int,
        default=512,
        help="Calibration input resolution (square). Use 256 or 384 to reduce RAM",
    )
    parser.add_argument("--method", type=str, default="entropy", choices=["entropy", "minmax"], help="Calibration method")
    parser.add_argument("--per-channel", action="store_true", help="Enable per-channel quantization where supported")
    parser.add_argument("--format", type=str, default="qdq", choices=["qdq", "qlinear"], help="Quant format: QDQ or QOperator (QLinear*)")
    parser.add_argument("--dynamic", action="store_true", help="Use dynamic (weight-only) quantization to skip calibration")
    # Performance/memory controls for ORT during calibration
    parser.add_argument("--threads", type=int, default=1, help="ORT intra-op threads (1 can reduce memory fragmentation)")
    parser.add_argument("--sequential", action="store_true", help="Force sequential execution (may lower peak memory)")
    parser.add_argument("--disable-mem-pattern", action="store_true", help="Disable memory pattern allocations")
    parser.add_argument("--disable-mem-arena", action="store_true", help="Disable CPU memory arena")
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
        help="Op types to quantize. Default focuses on Conv and MatMul to save memory",
    )
    parser.add_argument(
        "--split-optypes",
        action="store_true",
        help="Quantize each op type in separate passes to lower peak memory",
    )
    args = parser.parse_args()

    if args.dynamic:
        # Weight-only quantization; activations remain fp32
        quantize_dynamic(
            model_input=str(args.input_model),
            model_output=str(args.output_model),
            weight_type=QuantType.QInt8,
            per_channel=args.per_channel,
            op_types_to_quantize=["Conv", "MatMul"],
        )
        print(f"Saved dynamically-quantized (INT8 weights) model to: {args.output_model}")
        return

    # Configure ORT session options to lower memory pressure during calibration inference
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
    # Additional memory behavior tweaks (best-effort; ignore if unsupported)
    try:
        so.add_session_config_entry("session.enable_mem_reuse", "0")
        so.add_session_config_entry("session.intra_op_allow_spinning", "0")
        so.add_session_config_entry("session.set_denormal_as_zero", "1")
    except Exception:
        pass

    # Optionally reduce calibration spatial size to mitigate OOM
    global CALIB_IMAGE_SIZE
    calib_hw = max(32, int(args.calib_size))
    CALIB_IMAGE_SIZE = (calib_hw, calib_hw)

    input_name = discover_model_input_name(args.input_model)
    reader = FaceFolderDataReader(input_name, args.calib_dir, args.num_calib)

    method = CalibrationMethod.Entropy if args.method == "entropy" else CalibrationMethod.MinMax
    quant_format = QuantFormat.QDQ if args.format == "qdq" else QuantFormat.QOperator

    # Prefer QDQ format; pass optional kwargs when supported by installed onnxruntime version
    qs_sig = inspect.signature(quantize_static)
    # Base kwargs shared across passes
    base_kwargs = dict(
        calibration_data_reader=reader,
        quant_format=quant_format,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=method,
        per_channel=args.per_channel,
    )
    if "session_options" in qs_sig.parameters:
        base_kwargs["session_options"] = so
    if "providers" in qs_sig.parameters:
        base_kwargs["providers"] = ["CPUExecutionProvider"]
    if "provider_options" in qs_sig.parameters:
        base_kwargs["provider_options"] = None

    # Determine op-type groups (single group or split by type)
    op_types: List[str] = list(args.ops)
    groups: List[List[str]] = [[t] for t in op_types] if args.split_optypes else [op_types]

    # Run in passes to reduce instrumentation size and peak memory
    current_input = str(args.input_model)
    final_output = str(args.output_model)
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, group in enumerate(groups):
            is_last = idx == (len(groups) - 1)
            out_path = final_output if is_last else os.path.join(tmpdir, f"pass_{idx+1}.onnx")
            # Calibrator consumes the iterator; rewind before each pass
            reader.rewind()
            qs_kwargs = dict(base_kwargs)
            qs_kwargs["model_input"] = current_input
            qs_kwargs["model_output"] = out_path
            if "op_types_to_quantize" in qs_sig.parameters:
                qs_kwargs["op_types_to_quantize"] = group
            # Try quantization; if DequantizeLinear axis is unsupported by ORT, retry without per-channel and/or with QOperator
            try:
                quantize_static(**qs_kwargs)
            except Exception as e:
                msg = str(e)
                needs_retry_no_per_channel = "DequantizeLinear" in msg and "axis" in msg and base_kwargs.get("per_channel", False)
                if needs_retry_no_per_channel:
                    print("Warning: ORT does not support axis on DequantizeLinear in this environment. Retrying without per-channel.")
                    reader.rewind()
                    qs_kwargs["per_channel"] = False
                    base_kwargs["per_channel"] = False
                    try:
                        quantize_static(**qs_kwargs)
                        current_input = out_path
                        continue
                    except Exception:
                        pass
                # Retry with QOperator format if we were using QDQ
                if base_kwargs.get("quant_format") == QuantFormat.QDQ:
                    print("Retrying with QOperator (QLinear*) format for compatibility...")
                    reader.rewind()
                    qs_kwargs["quant_format"] = QuantFormat.QOperator
                    base_kwargs["quant_format"] = QuantFormat.QOperator
                    quantize_static(**qs_kwargs)
            current_input = out_path

    print(f"Saved statically-quantized (INT8) model to: {final_output}")


if __name__ == "__main__":
    main()


