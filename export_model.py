#!/usr/bin/env python3
"""
Export Hugging Face BiRefNet-portrait to ONNX (FP16).

- Loads pretrained weights directly from the HF Hub.
- Exports an FP32 ONNX first (more robust for exporters),
  then converts to FP16 using onnxconverter-common.
- Uses dynamic spatial dims by default (N, H, W).

Usage:
  python export_birefnet_hf_to_onnx_fp16.py \
      --onnx-out BiRefNet-portrait.fp16.onnx \
      --height 1024 --width 1024 --opset 17
"""

import argparse
from pathlib import Path

import torch
import onnx
from onnxconverter_common import float16

# transformers is only needed to load the model
from transformers import AutoModelForImageSegmentation
import deform_conv2d_onnx_exporter

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-id",
        type=str,
        default="ZhengPeng7/BiRefNet-portrait",
        help="Hugging Face model id",
    )
    ap.add_argument(
        "--onnx-out",
        default="D:/data/BiRefNet-portrait-epoch_150.fp16.onnx",
        type=str,
        help="Output ONNX filename (final FP16 model)",
    )
    ap.add_argument(
        "--height",
        type=int,
        default=512,
        help="Dummy export height (graph will be dynamic by default)",
    )
    ap.add_argument(
        "--width",
        type=int,
        default=512,
        help="Dummy export width (graph will be dynamic by default)",
    )
    ap.add_argument(
        "--opset",
        type=int,
        default=19,
        help="ONNX opset version (>=13 recommended; 17 is a good default)",
    )
    ap.add_argument(
        "--no-dynamic",
        action="store_true",
        help="Disable dynamic axes; fix the ONNX to the dummy HxW",
    )
    ap.add_argument(
        "--keep-io-fp32",
        action="store_true",
        help="Keep input/output tensors in FP32 when converting the graph to FP16",
    )
    return ap.parse_args()


def load_hf_model(model_id: str):
    print(f"[info] Loading {model_id} from Hugging Face (pretrained weights) …")
    model = AutoModelForImageSegmentation.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    return model


@torch.no_grad()
def export_fp32_then_fp16(model, out_path: str, height: int, width: int, opset: int, dynamic: bool, keep_io_fp32: bool):
    model.eval()
    # Exporters are most stable with FP32 dummy inputs
    dummy = torch.randn(1, 3, height, width, dtype=torch.float32)

    input_layer_names = ['input_image']
    output_layer_names = ['output_image']
    tmp_fp32 = Path(out_path).with_suffix(".fp32.onnx")
    deform_conv2d_onnx_exporter.register_deform_conv2d_onnx_op()
    print("[info] Exporting FP32 ONNX …")
    torch.onnx.export(
        model,
        dummy,
        tmp_fp32,
        verbose=False,
        opset_version=opset,
        input_names=input_layer_names,
        output_names=output_layer_names,
    )
    print(f"[ok] FP32 ONNX exported: {tmp_fp32}")

    print("[info] Converting ONNX to FP16 …")
    model_onnx = onnx.load(str(tmp_fp32))
    fp16_model = float16.convert_float_to_float16(
        model_onnx,
        keep_io_types=keep_io_fp32,  # True -> FP16 weights with FP32 I/O; False -> FP16 I/O too
    )
    onnx.save(fp16_model, out_path)
    print(f"[ok] FP16 ONNX saved: {out_path}")



def main():
    args = get_args()
    if args.opset < 13:
        raise SystemExit("[error] Use opset >= 13 (17 recommended).")

    model = load_hf_model(args.model_id)
    export_fp32_then_fp16(
        model=model,
        out_path=args.onnx_out,
        height=args.height,
        width=args.width,
        opset=args.opset,
        dynamic=(not args.no_dynamic),
        keep_io_fp32=args.keep_io_fp32,
    )
    print("\n[done] ONNX FP16 export complete. Use the same pre/post-processing as HF inference.")


if __name__ == "__main__":
    main()
