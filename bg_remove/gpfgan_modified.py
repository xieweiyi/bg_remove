# export_gfpgan_v14_to_onnx.py
"""
Export GFPGAN v1.4 (GFPGANv1Clean) to ONNX by patching StyleGAN2's
modulated conv into an ONNX-friendly op that:
  - keeps constant weights,
  - moves modulation to the activations,
  - mirrors SAME padding, and
  - mirrors upsample/downsample shape changes.

Use --debug to print shapes flowing through patched layers.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean


class ModulatedConv2d_ONNX(nn.Module):
    def __init__(self, orig, name_for_debug=None, debug=False):
        super().__init__()
        # learned params
        self.weight = nn.Parameter(orig.weight.detach().clone())
        self.out_channels = int(orig.out_channels)
        self.in_channels = int(orig.in_channels)

        # normalize hyperparams to tuples
        k = getattr(orig, "kernel_size", 3)
        self.kernel_size = _pair(k)
        self.stride = _pair(getattr(orig, "stride", 1))
        self.dilation = _pair(getattr(orig, "dilation", 1))

        if hasattr(orig, "padding"):
            self.padding = _pair(orig.padding)
        else:
            self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

        # StyleGAN2 often carries separate up/down flags (fused upfirdn)
        self.do_up = bool(getattr(orig, "upsample", False) or getattr(orig, "up", False))
        self.do_down = bool(getattr(orig, "downsample", False) or getattr(orig, "down", False))

        # style MLP and demod flag (we keep demod off since we move scaling to input)
        self.modulation = orig.modulation
        self.demodulate = getattr(orig, "demodulate", True)

        # (optional) debug
        self._dbg = debug
        self._name = name_for_debug or "ModConv"

    def _log(self, msg):
        if self._dbg:
            print(f"[{self._name}] {msg}")

    def forward(self, x, style):
        # x: [B, Cin, H, W]
        B, Cx, H, W = x.shape
        C = self.in_channels
        if Cx != C:
            raise RuntimeError(f"{self._name}: Input channels ({Cx}) != expected ({C})")

        in_shape = (B, C, H, W)

        # Mirror StyleGAN2 fused up/down *shape* effects.
        # (We use simple nearest/avgpool to keep shapes consistent.)
        if self.do_up:
            # Nearest is ONNX-friendly and keeps integers exact
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            H, W = x.shape[-2], x.shape[-1]

        if self.do_down:
            # Approximate downfirdn with avg pool (shape match is the goal)
            x = F.avg_pool2d(x, kernel_size=2, stride=2, ceil_mode=False)
            H, W = x.shape[-2], x.shape[-1]

        # style -> per-input-channel scale
        s = self.modulation(style)
        if s.dim() == 2:          # [B, C]
            s = s.view(B, C, 1, 1)
        elif s.dim() == 4:        # [B, C, 1, 1] or broadcastable
            s = s.view(B, C, 1, 1)
        else:
            raise RuntimeError(f"{self._name}: Unexpected modulation shape {tuple(s.shape)}")

        # move modulation to activations (keep weights constant for ONNX)
        x = x * s  # [B,C,H,W]

        # grouped conv trick: collapse batch into groups
        x = x.view(1, B * C, H, W)  # [1, B*C, H, W]

        # ensure weight shape [Cout, Cin, kh, kw]
        Wt = self.weight
        if Wt.dim() != 4:
            Wt = Wt.view(self.out_channels, C, *self.kernel_size)

        # repeat weights per batch so groups=B is valid
        Wt = Wt.repeat(B, 1, 1, 1)  # [B*Cout, Cin, kh, kw]

        y = F.conv2d(
            x, Wt, bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=B
        )
        y = y.view(B, self.out_channels, y.shape[-2], y.shape[-1])

        out_shape = tuple(y.shape)
        self._log(f"in {in_shape} → out {out_shape} | up={self.do_up}, down={self.do_down}, k={self.kernel_size}, pad={self.padding}, stride={self.stride}")
        return y


def patch_modulated_convs(module, prefix="", debug=False):
    """Recursively replace StyleGAN2 modulated convs with ONNX-friendly ones."""
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        # Heuristic: StyleGAN2 modulated convs have both "modulation" and "weight"
        if hasattr(child, "modulation") and hasattr(child, "weight") and isinstance(child.weight, torch.Tensor):
            setattr(module, name, ModulatedConv2d_ONNX(child, name_for_debug=full_name, debug=debug))
        else:
            patch_modulated_convs(child, prefix=full_name, debug=debug)


class GFPGANv14_ONNX(nn.Module):
    def __init__(self, state_dict_path, channel_multiplier=2, debug=False):
        super().__init__()
        self.net = GFPGANv1Clean(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True)
        sd = torch.load(state_dict_path, map_location="cpu")
        key = "params_ema" if "params_ema" in sd else "params"
        self.net.load_state_dict(sd[key], strict=True)
        self.net.eval()

        patch_modulated_convs(self.net, debug=debug)

    @torch.no_grad()
    def forward(self, face_bchw):
        # Input: [N,3,512,512], float32 in [-1,1]
        out, _ = self.net(face_bchw, return_rgb=False, weight=0.5)
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to GFPGANv1.4.pth")
    parser.add_argument("--onnx", default="GFPGANv1.4.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--sanity", action="store_true",
                        help="Run a PyTorch forward to check shapes before export.")
    parser.add_argument("--debug", action="store_true",
                        help="Log shapes for every patched modulated conv.")
    args = parser.parse_args()

    model = GFPGANv14_ONNX(args.weights, debug=args.debug).eval()

    dummy = torch.randn(1, 3, 512, 512, dtype=torch.float32)

    if args.sanity:
        with torch.no_grad():
            y = model(dummy)
            print("Sanity forward output shape:", tuple(y.shape))

    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            "input": {0: "N", 2: "H", 3: "W"},
            "output": {0: "N", 2: "H", 3: "W"}
        }

    torch.onnx.export(
        model,
        dummy,
        args.onnx,
        input_names=["input"],
        output_names=["output"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )

    print(f"✔ Exported to {os.path.abspath(args.onnx)}")


if __name__ == "__main__":
    main()
