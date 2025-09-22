#!/usr/bin/env python3
"""
Compare BiRefNet-lite ONNX FP32 vs INT8 on a folder of images.

- Preprocessing: RGB -> resize 512x512 -> float32 -> normalize by ImageNet mean/std -> NCHW
- Runs both models via ONNX Runtime on CPU
- For each source image, saves ONE side-by-side PNG to the output dir named after the source file:
  left = original with background set to white using FP32 matte; right = INT8

Usage:
  python bg_remove/compare_birefnet_int8_vs_fp32.py \
    --fp32 d:/data/models/BiRefNet_lite-matting.onnx \
    --int8 data/models/BiRefNet_lite-matting-int8.onnx \
    --images d:/data/PPM-100/image \
    --out out/compare_birefnet --limit 50
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import onnx
import onnxruntime as ort


IMAGE_SIZE = (512, 512)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def list_images(root: Path, exts: Optional[List[str]] = None) -> List[Path]:
    exts = exts or [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


def discover_io(model_path: Path) -> Tuple[str, List[str]]:
    model = onnx.load(str(model_path))
    graph = model.graph
    if not graph.input:
        raise RuntimeError("Model has no inputs")
    input_name = graph.input[0].name
    output_names = [o.name for o in graph.output]
    return input_name, output_names


def create_session(model_path: Path, threads: int = 1) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, int(threads))
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.enable_mem_pattern = False
    so.enable_cpu_mem_arena = True
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"], sess_options=so)
    return sess


def preprocess(img_path: Path) -> Tuple[np.ndarray, Image.Image]:
    img = Image.open(img_path).convert("RGB")
    img_resized = img.resize(IMAGE_SIZE, Image.LANCZOS)
    arr = np.asarray(img_resized).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    x = arr.transpose(2, 0, 1)[None]
    return x, img_resized


def _apply_activation(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return y
    if mode == "sigmoid":
        return 1.0 / (1.0 + np.exp(-y))
    if mode == "tanh":
        return (np.tanh(y) + 1.0) * 0.5
    if mode == "div255":
        return y / 255.0
    if mode == "clip01":
        return np.clip(y, 0.0, 1.0)
    # auto
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    # Heuristics
    if y_max > 6.0 or y_min < -6.0:
        return 1.0 / (1.0 + np.exp(-y))
    if y_min >= -1.1 and y_max <= 1.1 and (y_min < 0.0) and (y_max > 0.0):
        return (np.tanh(y) + 1.0) * 0.5
    if 1.5 < y_max <= 255.5 and y_min >= 0.0:
        return y / 255.0
    if y_min < 0.0 or y_max > 1.0:
        # Conservative: clip into [0,1]
        return np.clip(y, 0.0, 1.0)
    return y


def postprocess_matte(output_tensors: List[np.ndarray], activation: str) -> np.ndarray:
    # Try typical matte shapes: (1,1,H,W) or (1,H,W) or (1, H, W, 1)
    y = None
    for t in output_tensors:
        if t.ndim == 4 and t.shape[1] == 1:
            y = t[0, 0]
            break
        if t.ndim == 3:
            y = t[0]
            break
        if t.ndim == 4 and t.shape[-1] == 1:
            y = t[0, ..., 0]
            break
    if y is None:
        # Fallback to first output
        y = output_tensors[0].squeeze()
    y = np.asarray(y, dtype=np.float32)
    y = _apply_activation(y, activation)
    return np.clip(y, 0.0, 1.0)


def maybe_auto_invert(alpha01: np.ndarray, enable: bool) -> np.ndarray:
    if not enable:
        return alpha01
    h, w = alpha01.shape
    border = 0.1
    bw = max(1, int(w * border))
    bh = max(1, int(h * border))
    # Border region mean (likely background), center region mean (likely foreground)
    border_mean = float(np.mean(np.concatenate([
        alpha01[:bh, :], alpha01[-bh:, :], alpha01[:, :bw], alpha01[:, -bw:]
    ], axis=None)))
    center_mean = float(np.mean(alpha01[bh:h-bh, bw:w-bw])) if (h - 2*bh > 1 and w - 2*bw > 1) else float(np.mean(alpha01))
    # If border is more foreground-like (higher) than center, invert
    if border_mean - center_mean > 0.05:
        return 1.0 - alpha01
    return alpha01


def blend_over_background(original_rgb: Image.Image, alpha01_resized: np.ndarray, bg_rgb: Tuple[int, int, int]) -> Image.Image:
    # Upscale matte to original size
    a8_small = (np.clip(alpha01_resized, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    a_big = Image.fromarray(a8_small, mode="L").resize(original_rgb.size, Image.BILINEAR)
    # Foreground = alpha * original + (1-alpha) * bg
    fg = original_rgb.convert("RGB")
    bg = Image.new("RGB", original_rgb.size, bg_rgb)
    a = (np.asarray(a_big).astype(np.float32) / 255.0)[..., None]
    fg_np = np.asarray(fg).astype(np.float32)
    bg_np = np.asarray(bg).astype(np.float32)
    out = a * fg_np + (1.0 - a) * bg_np
    out_u8 = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(out_u8, mode="RGB")


def to_uint8_gray(alpha01: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(alpha01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="L")


def compute_metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    a = np.clip(a, 0.0, 1.0)
    b = np.clip(b, 0.0, 1.0)
    diff = a - b
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff * diff))
    psnr = 99.0 if mse == 0 else float(20.0 * math.log10(1.0 / math.sqrt(mse)))
    return {"MAE": mae, "MSE": mse, "PSNR": psnr}


def save_side_by_side_image(original_rgb: Image.Image, matte32: np.ndarray, matte8: np.ndarray, bg_rgb: Tuple[int, int, int], out_path: Path) -> None:
    left = blend_over_background(original_rgb, matte32, bg_rgb)
    right = blend_over_background(original_rgb, matte8, bg_rgb)
    w, h = left.size
    canvas = Image.new("RGB", (w * 2, h))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (w, 0))
    canvas.save(out_path)


def _parse_bg_color(s: str) -> Tuple[int, int, int]:
    s = s.strip()
    named = {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "gray": (128, 128, 128),
        "grey": (128, 128, 128),
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
    }
    if s.lower() in named:
        return named[s.lower()]
    if s.startswith("#") and (len(s) == 7):
        r = int(s[1:3], 16)
        g = int(s[3:5], 16)
        b = int(s[5:7], 16)
        return (r, g, b)
    if "," in s:
        parts = s.split(",")
        if len(parts) == 3:
            r, g, b = [max(0, min(255, int(p))) for p in parts]
            return (r, g, b)
    raise ValueError(f"Unrecognized bg color: {s}")


def run(args):
    images = list_images(Path(args.images))
    if args.limit is not None:
        images = images[: args.limit]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    fp32_in, _ = discover_io(Path(args.fp32))
    int8_in, _ = discover_io(Path(args.int8))

    sess_fp32 = create_session(Path(args.fp32), threads=args.threads)
    sess_int8 = create_session(Path(args.int8), threads=args.threads)

    bg_rgb = _parse_bg_color(args.bg)

    for img_path in images:
        # Keep original for final compositing, but preprocess resized for model
        original = Image.open(img_path).convert("RGB")
        x, _resized = preprocess(img_path)

        y32_list = sess_fp32.run(None, {fp32_in: x})
        y8_list = sess_int8.run(None, {int8_in: x})
        matte32 = postprocess_matte(y32_list, args.matte_activation)
        matte8 = postprocess_matte(y8_list, args.matte_activation)
        # First optional manual invert, then auto-invert if enabled
        if args.invert_alpha:
            matte32 = 1.0 - matte32
            matte8 = 1.0 - matte8
        matte32 = maybe_auto_invert(matte32, args.auto_invert)
        matte8 = maybe_auto_invert(matte8, args.auto_invert)

        out_name = f"{img_path.stem}_cmp.png"
        save_side_by_side_image(original, matte32, matte8, bg_rgb, out_dir / out_name)
    print("Done. Side-by-side results in:", str(out_dir))


def main():
    ap = argparse.ArgumentParser(description="Compare BiRefNet ONNX FP32 vs INT8")
    ap.add_argument("--fp32", type=str, required=True, help="Path to FP32 ONNX model")
    ap.add_argument("--int8", type=str, required=True, help="Path to INT8 ONNX model")
    ap.add_argument("--images", type=str, required=True, help="Folder of images to process")
    ap.add_argument("--out", type=str, default="out/compare_birefnet", help="Output directory")
    ap.add_argument("--threads", type=int, default=1, help="ORT threads")
    ap.add_argument("--invert-alpha", action="store_true", help="Invert the predicted matte (use if foreground/background are swapped)")
    ap.add_argument("--auto-invert", action="store_true", help="Auto-detect and correct matte polarity via border-vs-center heuristic")
    ap.add_argument(
        "--matte-activation",
        type=str,
        default="auto",
        choices=["auto", "sigmoid", "tanh", "div255", "clip01", "none"],
        help="How to map model output to [0,1] alpha",
    )
    ap.add_argument("--bg", type=str, default="255,255,255", help="Background color: name (white), #RRGGBB, or R,G,B")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of images")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()


